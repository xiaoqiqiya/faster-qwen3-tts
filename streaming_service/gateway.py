from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, HTTPException, Request

from .cache import CapacityError, GenerationError, SharedCache, request_key
from .common import (AUDIO, DONE, ERROR, AudioResponse, FrameDecoder, Limits, SpeechRequest,
                     envelope, error_frame, frame, install_errors, resolve_request, HEADER,
                     identity_version, PROTOCOL)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
LIMITS = Limits.from_env()
BACKEND_URL = os.getenv("TTS_BACKEND_URL", "http://backend:8001").rstrip("/")


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(timeout=httpx.Timeout(LIMITS.task_seconds, connect=5),
                                trust_env=False) as client:
        app.state.client = client
        app.state.info = None
        app.state.cache = SharedCache(LIMITS, backend_audio)
        sweep = asyncio.create_task(app.state.cache.sweep())
        try:
            yield
        finally:
            sweep.cancel()
            with suppress(asyncio.CancelledError):
                await sweep
            await app.state.cache.close()


app = FastAPI(title="Qwen3-TTS 流式网关", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
install_errors(app)


async def backend_info():
    try:
        response = await app.state.client.get(BACKEND_URL + "/health", timeout=5)
        response.raise_for_status()
        info = response.json()["data"]
        if info["max_output_bytes"] != LIMITS.output_bytes or info["task_timeout_seconds"] != LIMITS.task_seconds:
            raise ValueError("网关和推理服务的任务限制不一致")
        app.state.info = info
        return info
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("推理服务未就绪: %s", exc)
        raise HTTPException(503, "推理服务暂不可用或配置不一致") from exc


async def backend_audio(req, expected_key):
    decoder = FrameDecoder()
    info = await backend_info()
    if request_key(req, info) != expected_key:
        raise GenerationError("排队期间模型或参考音色版本已变更")
    params = req.model_copy(update={"response_format": "pcm_framed"}).model_dump()
    async with app.state.client.stream("POST", BACKEND_URL + "/v1/audio/speech", json=params,
                                       headers={"X-TTS-Backend-Version": identity_version(info)}) as response:
        if response.status_code != 200:
            raise GenerationError("推理服务拒绝请求")
        if response.headers.get("X-Audio-Protocol") != PROTOCOL:
            raise GenerationError("推理服务的音频协议不一致")
        if response.headers.get("X-Audio-Sample-Rate") != str(info["sample_rate"]):
            raise GenerationError("推理服务的音频采样率不一致")
        async for block in response.aiter_bytes(chunk_size=None):
            for kind, payload in decoder.feed(block):
                if kind == ERROR:
                    raise GenerationError("推理服务报告流中失败")
                if kind == AUDIO:
                    yield payload
        decoder.finish()


@app.get("/health")
async def health():
    info = await backend_info()
    cache = app.state.cache
    return envelope(200, "流式网关已就绪", {
        "backend": info, "cache_entries": len(cache.records), "active_readers": cache.active_readers,
        "active_tasks": len(cache.producers), "cache_accounted_bytes": cache.accounted_bytes,
    })


@app.get("/")
async def root():
    return envelope(200, "Qwen3-TTS 流式网关", {"health": "/health", "speech": "/v1/audio/speech"})


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, request: Request):
    allow_new = True
    try:
        info = await backend_info()
    except HTTPException:
        info = app.state.info
        if info is None:
            raise
        allow_new = False
    req = resolve_request(req, info)
    key = request_key(req, info)
    cache = app.state.cache
    try:
        record = await cache.acquire(key, req, allow_new=allow_new)
    except CapacityError as exc:
        raise HTTPException(503, str(exc)) from exc
    released = False

    async def release():
        nonlocal released
        if not released:
            released = True
            await cache.release(record)

    async def wait_first():
        while not record.chunks and not record.done:
            if await request.is_disconnected():
                raise HTTPException(400, "客户端连接已关闭")
            try:
                await asyncio.wait_for(cache.first(record), 0.25)
            except asyncio.TimeoutError:
                continue
        await cache.first(record)

    try:
        await wait_first()
    except GenerationError as exc:
        await release()
        raise HTTPException(503, str(exc)) from exc
    except BaseException:
        await release()
        raise

    async def body():
        first = True
        try:
            async for chunk in cache.read(record):
                if first:
                    logger.info("网关发送首块 key=%s monotonic=%.6f", key, time.monotonic())
                    first = False
                if req.response_format == "pcm_framed":
                    yield HEADER.pack(AUDIO, len(chunk))
                yield chunk
            if req.response_format == "pcm_framed":
                yield frame(DONE)
        except GenerationError as exc:
            logger.warning("客户端音频流失败 key=%s message=%s", key, exc)
            if req.response_format == "pcm_framed":
                yield error_frame(str(exc))
            else:
                # Raising aborts the HTTP body rather than sending a clean end of stream.
                raise

    return AudioResponse(body(), sample_rate=info["sample_rate"], fmt=req.response_format,
                         send_seconds=LIMITS.send_seconds, release=release)
