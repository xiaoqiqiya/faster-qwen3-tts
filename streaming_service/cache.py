from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from .common import Limits, SpeechRequest

logger = logging.getLogger(__name__)


class CapacityError(Exception):
    pass


class GenerationError(Exception):
    pass


@dataclass(eq=False)
class Record:
    key: str
    request: SpeechRequest
    created_at: float
    last_access_at: float
    chunks: list[bytes] = field(default_factory=list)
    done: bool = False
    error: str | None = None
    completed_at: float | None = None
    active_readers: int = 0
    size: int = 0
    accounted: int = 0
    producer: asyncio.Task | None = None
    first_chunk_at: float | None = None


def request_key(req: SpeechRequest, identity: dict) -> str:
    # The transport envelope does not change the generated PCM.
    params = req.model_dump(exclude={"response_format"})
    payload = json.dumps({"request": params, "backend": identity}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SharedCache:
    def __init__(self, limits: Limits, producer: Callable[[SpeechRequest, str], AsyncIterator[bytes]],
                 *, clock=time.monotonic):
        self.limits, self.produce, self.clock = limits, producer, clock
        self.records: dict[str, Record] = {}
        self.condition = asyncio.Condition()
        self.gpu_slot = asyncio.Semaphore(1)
        self.producers: set[asyncio.Task] = set()
        self.accounted_bytes = 0
        self.active_readers = 0
        self.closed = False

    def _drop(self, record: Record):
        if self.records.get(record.key) is record:
            del self.records[record.key]
        self.accounted_bytes -= record.accounted
        record.accounted = 0

    def _clean(self):
        now = self.clock()
        for record in list(self.records.values()):
            if record.done and not record.active_readers and (
                record.error or now - max(record.completed_at, record.last_access_at) >= self.limits.ttl
            ):
                self._drop(record)

    async def acquire(self, key: str, req: SpeechRequest, *, allow_new: bool = True):
        async with self.condition:
            self._clean()
            if self.closed or self.active_readers >= self.limits.readers:
                raise CapacityError("服务繁忙，读取连接数已达到上限")
            record = self.records.get(key)
            if record is None:
                if not allow_new:
                    raise CapacityError("推理服务暂不可用，且没有可用缓存")
                # At most one running producer plus the bounded waiting producers.
                if len(self.producers) >= self.limits.pending + 1:
                    raise CapacityError("服务繁忙，待生成任务已达到上限")
                candidates = sorted((r for r in self.records.values() if r.done and not r.active_readers),
                                    key=lambda r: r.last_access_at)
                for candidate in candidates:
                    if self.accounted_bytes + self.limits.output_bytes <= self.limits.cache_bytes:
                        break
                    self._drop(candidate)
                if self.accounted_bytes + self.limits.output_bytes > self.limits.cache_bytes:
                    raise CapacityError("服务繁忙，音频缓存容量不足")
                now = self.clock()
                record = Record(key, req, now, now, accounted=self.limits.output_bytes)
                self.records[key] = record
                self.accounted_bytes += record.accounted
                record.producer = asyncio.create_task(self._run(record))
                self.producers.add(record.producer)
                record.producer.add_done_callback(self.producers.discard)
            record.last_access_at = self.clock()
            record.active_readers += 1
            self.active_readers += 1
            return record

    async def release(self, record: Record):
        async with self.condition:
            record.active_readers -= 1
            self.active_readers -= 1
            if record.done and record.error and not record.active_readers:
                self._drop(record)
            self._clean()

    async def _generate(self, record: Record):
        async with self.gpu_slot:
            logger.info("生成开始 key=%s queued_ms=%.1f", record.key, (self.clock() - record.created_at) * 1000)
            iterator = self.produce(record.request, record.key)
            try:
                async for chunk in iterator:
                    if not isinstance(chunk, bytes) or not chunk or len(chunk) % 2:
                        raise GenerationError("推理服务返回无效 PCM")
                    async with self.condition:
                        if record.size + len(chunk) > self.limits.output_bytes:
                            raise GenerationError("生成音频超过单任务输出上限")
                        if not record.chunks:
                            record.first_chunk_at = self.clock()
                            logger.info("共享任务收到首块 key=%s elapsed_ms=%.1f", record.key,
                                        (record.first_chunk_at - record.created_at) * 1000)
                        record.chunks.append(chunk)
                        record.size += len(chunk)
                        self.condition.notify_all()
            finally:
                await iterator.aclose()
            if not record.size:
                raise GenerationError("推理服务没有生成音频")

    async def _run(self, record: Record):
        error = None
        try:
            remaining = self.limits.task_seconds - (self.clock() - record.created_at)
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(self._generate(record), remaining)
        except asyncio.CancelledError:
            error = "服务关闭，生成已中止"
        except asyncio.TimeoutError:
            error = "任务排队或生成超过总耗时上限"
        except Exception:
            logger.exception("生成失败 key=%s", record.key)
            error = "语音生成失败，请重试"
        finally:
            async with self.condition:
                record.done, record.error, record.completed_at = True, error, self.clock()
                self.accounted_bytes -= record.accounted - record.size
                record.accounted = record.size
                if error:
                    if self.records.get(record.key) is record:
                        del self.records[record.key]
                    if not record.active_readers:
                        self._drop(record)
                self.condition.notify_all()
            logger.info("生成结束 key=%s success=%s bytes=%d elapsed_ms=%.1f", record.key,
                        error is None, record.size, (self.clock() - record.created_at) * 1000)

    async def first(self, record: Record):
        async with self.condition:
            await self.condition.wait_for(lambda: bool(record.chunks) or record.done)
            if record.error:
                raise GenerationError(record.error)
            if not record.chunks:
                raise GenerationError(record.error or "没有生成音频")

    async def read(self, record: Record):
        index = 0
        while True:
            async with self.condition:
                await self.condition.wait_for(lambda: index < len(record.chunks) or record.done)
                if index < len(record.chunks):
                    chunk = record.chunks[index]
                    index += 1
                elif record.error:
                    raise GenerationError(record.error)
                else:
                    return
            yield chunk

    async def sweep(self):
        while True:
            await asyncio.sleep(self.limits.cleanup_seconds)
            async with self.condition:
                self._clean()

    async def close(self):
        self.closed = True
        tasks = list(self.producers)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
