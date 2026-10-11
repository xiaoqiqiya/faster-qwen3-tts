from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import struct
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)
HEADER = struct.Struct("!BI")
AUDIO, DONE, ERROR = 1, 2, 3
MAX_FRAME_BYTES = 262144
PROTOCOL = "qwen-pcm-v1"


def envelope(code: int, message: str, data=None):
    return {"code": code, "message": message, "data": data}


def identity_version(info: dict):
    return hashlib.sha256(json.dumps(info, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def install_errors(app: FastAPI):
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "请求处理失败"
        if exc.status_code == 404 and exc.detail == "Not Found":
            message = "资源不存在"
        elif exc.status_code == 405 and exc.detail == "Method Not Allowed":
            message = "请求方法不允许"
        return JSONResponse(envelope(exc.status_code, message), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(map(str, e["loc"])), "type": e["type"]}
                   for e in exc.errors()]
        return JSONResponse(envelope(400, "请求参数校验失败", details), status_code=400)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception):
        logger.exception("请求处理失败", exc_info=exc)
        return JSONResponse(envelope(500, "服务端处理失败"), status_code=500)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input: str = Field(min_length=1, max_length=2048)
    voice: str | None = Field(default=None, min_length=1, max_length=128)
    language: str = Field(default="Chinese", min_length=1, max_length=32)
    instruct: str = Field(default="", max_length=512)
    speed: float = 1.0
    response_format: Literal["pcm", "pcm_framed"] = "pcm_framed"


def resolve_request(req: SpeechRequest, info: dict) -> SpeechRequest:
    if not req.input.strip():
        raise HTTPException(400, "合成文本不能为空")
    if req.speed != 1.0:
        raise HTTPException(400, "当前流式后端只支持语速 1.0")
    voice = req.voice or info["default_voice"]
    if voice not in info["voices"]:
        raise HTTPException(404, "指定音色不存在")
    if req.language.lower() not in [v.lower() for v in info["languages"]]:
        raise HTTPException(400, "指定语言不受支持")
    if req.instruct and not info["supports_instruct"]:
        raise HTTPException(400, "当前模型不支持语音指令，请使用支持指令的模型")
    language = next(v for v in info["languages"] if v.lower() == req.language.lower())
    return req.model_copy(update={"voice": voice, "language": language})


@dataclass(frozen=True)
class Limits:
    ttl: float = 300
    cache_bytes: int = 256 * 1024 * 1024
    output_bytes: int = 8 * 1024 * 1024
    pending: int = 8
    readers: int = 256
    task_seconds: float = 120
    send_seconds: float = 15
    cleanup_seconds: float = 5

    @classmethod
    def from_env(cls):
        values = {
            "ttl": ("TTS_CACHE_TTL_SECONDS", float),
            "readers": ("TTS_MAX_READERS", int),
            "task_seconds": ("TTS_TASK_TIMEOUT_SECONDS", float),
            "send_seconds": ("TTS_SEND_TIMEOUT_SECONDS", float),
            "cleanup_seconds": ("TTS_CACHE_CLEANUP_SECONDS", float),
        }
        defaults = cls()
        settings = {key: cast(os.getenv(env, str(getattr(defaults, key))))
                    for key, (env, cast) in values.items()}
        # Configuration uses MB (1024 * 1024 bytes); runtime accounting uses bytes.
        mb = 1024 * 1024
        settings["cache_bytes"] = int(os.getenv("TTS_CACHE_MAX_MB", str(defaults.cache_bytes // mb))) * mb
        settings["output_bytes"] = min(defaults.output_bytes, settings["cache_bytes"])
        result = cls(**settings)
        import math
        if any(not math.isfinite(v) or v <= 0 for v in result.__dict__.values()):
            raise ValueError("所有资源限制必须是有限正数")
        if result.output_bytes > result.cache_bytes or result.output_bytes % 2:
            raise ValueError("单任务输出限制必须是偶数，且不能超过缓存容量")
        if os.getenv("WORKERS", "1") != "1":
            raise ValueError("共享任务服务只允许 WORKERS=1")
        return result


def frame(kind: int, payload: bytes = b"") -> bytes:
    if kind not in (AUDIO, DONE, ERROR) or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("无效音频帧")
    return HEADER.pack(kind, len(payload)) + payload


class FrameDecoder:
    """HTTP read boundaries may split a header, payload, or PCM sample."""

    def __init__(self):
        self.buffer = bytearray()
        self.terminal = False

    def feed(self, block: bytes):
        self.buffer.extend(block)
        while len(self.buffer) >= HEADER.size:
            if self.terminal:
                raise ValueError("终止帧后出现额外数据")
            kind, size = HEADER.unpack_from(self.buffer)
            if kind not in (AUDIO, DONE, ERROR) or size > MAX_FRAME_BYTES:
                raise ValueError("无效音频帧头")
            if kind == AUDIO and (not size or size % 2):
                raise ValueError("PCM 帧必须包含完整采样")
            if kind == DONE and size:
                raise ValueError("完成帧不得包含数据")
            if len(self.buffer) < HEADER.size + size:
                break
            payload = bytes(self.buffer[HEADER.size:HEADER.size + size])
            del self.buffer[:HEADER.size + size]
            self.terminal = kind in (DONE, ERROR)
            yield kind, payload
        if self.terminal and self.buffer:
            raise ValueError("终止帧后出现额外数据")

    def finish(self):
        if self.buffer or not self.terminal:
            raise ValueError("音频流中断或缺少显式终止帧")


def audio_headers(sample_rate: int, fmt: str):
    return {
        "Cache-Control": "no-store", "X-Accel-Buffering": "no",
        "X-Audio-Sample-Rate": str(sample_rate), "X-Audio-Channels": "1",
        "X-Audio-Sample-Format": "s16le", "X-Audio-Protocol": PROTOCOL if fmt == "pcm_framed" else "raw",
    }


class AudioResponse(StreamingResponse):
    def __init__(self, content, *, sample_rate: int, fmt: str, send_seconds: float, release=None):
        super().__init__(content, media_type="application/vnd.qwen.pcm-stream" if fmt == "pcm_framed" else "audio/pcm",
                         headers=audio_headers(sample_rate, fmt))
        self.send_seconds = send_seconds
        self.release = release

    async def __call__(self, scope, receive, send):
        async def bounded_send(message):
            await asyncio.wait_for(send(message), self.send_seconds)
        try:
            await super().__call__(scope, receive, bounded_send)
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                if self.release:
                    await asyncio.shield(self.release())


def error_frame(message: str):
    return frame(ERROR, json.dumps(envelope(500, message), ensure_ascii=False).encode("utf-8"))
