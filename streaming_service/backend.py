from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .cache import request_key
from .common import (AUDIO, DONE, AudioResponse, Limits, SpeechRequest, envelope, error_frame,
                     frame, identity_version, install_errors, resolve_request, HEADER)
from .inference import Runtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
LIMITS = Limits.from_env()
runtime = Runtime(LIMITS)


@asynccontextmanager
async def lifespan(_app):
    await asyncio.to_thread(runtime.load)
    yield
    if runtime.thread:
        await asyncio.to_thread(runtime.thread.join, LIMITS.task_seconds + 2)


app = FastAPI(title="Qwen3-TTS 推理服务", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
install_errors(app)


@app.get("/health")
async def health():
    if runtime.info is None:
        raise HTTPException(503, "模型尚未加载和预热完成")
    return envelope(200, "推理服务已就绪", runtime.info)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, request: Request):
    if runtime.info is None:
        raise HTTPException(503, "模型尚未就绪")
    req = resolve_request(req, runtime.info)
    expected_version = request.headers.get("X-TTS-Backend-Version")
    if expected_version and expected_version != identity_version(runtime.info):
        raise HTTPException(409, "模型或参考音色版本已变更，请重新提交请求")
    if runtime.lock.locked():
        raise HTTPException(409, "推理服务正在执行其他任务")
    iterator = runtime.stream(req, request_key(req, runtime.info))
    try:
        first = await anext(iterator)
    except StopAsyncIteration as exc:
        await iterator.aclose()
        raise HTTPException(500, "模型没有生成音频") from exc
    except Exception as exc:
        await iterator.aclose()
        logger.exception("首块推理失败")
        raise HTTPException(500, "语音生成失败，未生成可播放音频") from exc
    except BaseException:
        await iterator.aclose()
        raise

    async def body():
        try:
            yield HEADER.pack(AUDIO, len(first))
            yield first
            async for chunk in iterator:
                yield HEADER.pack(AUDIO, len(chunk))
                yield chunk
            yield frame(DONE)
        except Exception:
            logger.exception("流中推理失败")
            yield error_frame("推理失败或音频达到生成上限")

    return AudioResponse(body(), sample_rate=runtime.info["sample_rate"], fmt="pcm_framed",
                         send_seconds=LIMITS.send_seconds, release=iterator.aclose)
