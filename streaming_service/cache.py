from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable

from .common import Limits, SpeechRequest, MAX_FRAME_BYTES

logger = logging.getLogger(__name__)


class CapacityError(Exception):
    pass


class GenerationError(Exception):
    pass


@dataclass(eq=False)
class Record:
    key: str
    request: SpeechRequest | None
    created_at: float
    last_access_at: float
    path: Path
    done: bool = False
    error: str | None = None
    completed_at: float | None = None
    active_readers: int = 0
    size: int = 0
    accounted: int = 0
    producer: asyncio.Task | None = None
    first_chunk_at: float | None = None
    queued_at: float = field(default_factory=time.monotonic)


def request_key(req: SpeechRequest, identity: dict) -> str:
    # The transport envelope does not change the generated PCM.
    params = req.model_dump(exclude={"response_format"})
    payload = json.dumps({"request": params, "backend": identity}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SharedCache:
    def __init__(self, limits: Limits, producer: Callable[[SpeechRequest, str], AsyncIterator[bytes]],
                 *, cache_dir="cache/audio", clock=time.time):
        self.limits, self.produce, self.clock = limits, producer, clock
        self.directory = Path(cache_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, Record] = {}
        self.retired: set[Record] = set()
        self.condition = asyncio.Condition()
        self.gpu_slot = asyncio.Semaphore(1)
        self.producers: set[asyncio.Task] = set()
        self.accounted_bytes = 0
        self.active_readers = 0
        self.closed = False
        self._restore()

    async def _io(self, function, *args):
        # Finish filesystem work before cancellation can remove or reuse its files.
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _restore(self):
        now = self.clock()
        for metadata in self.directory.glob("tts-*.json"):
            path = metadata.with_suffix(".pcm")
            try:
                data = json.loads(metadata.read_text(encoding="utf-8"))
                size = data["size"]
                completed, accessed = data["completed_at"], data["last_access_at"]
                if (data["version"] != 1 or not isinstance(data["key"], str)
                    or not isinstance(size, int) or not 0 < size <= self.limits.output_bytes
                    or size % 2 or path.stat().st_size != size
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (completed, accessed))
                    or now - max(completed, accessed) >= self.limits.ttl):
                    raise ValueError("过期或无效的磁盘缓存")
                record = Record(data["key"], None, completed, accessed, path,
                                done=True, completed_at=completed, size=size, accounted=size)
                previous = self.records.get(record.key)
                if previous:
                    if previous.last_access_at >= accessed:
                        self._delete_files(record)
                        continue
                    self._drop(previous)
                self.records[record.key] = record
                self.accounted_bytes += size
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("清理无效磁盘缓存 path=%s", metadata)
                metadata.unlink(missing_ok=True)
                path.unlink(missing_ok=True)
        retained = {record.path for record in self.records.values()}
        for path in self.directory.glob("tts-*.pcm"):
            if path not in retained:
                path.unlink(missing_ok=True)
        for path in self.directory.glob("tts-*.tmp"):
            path.unlink(missing_ok=True)
        for record in sorted(self.records.values(), key=lambda r: r.last_access_at):
            if self.accounted_bytes <= self.limits.cache_bytes:
                break
            self._drop(record)

    @staticmethod
    def _delete_files(record):
        record.path.with_suffix(".json").unlink(missing_ok=True)
        record.path.with_suffix(".tmp").unlink(missing_ok=True)
        record.path.unlink(missing_ok=True)

    def _persist(self, record):
        temporary = record.path.with_suffix(".tmp")
        metadata = {"version": 1, "key": record.key, "size": record.size,
                    "completed_at": record.completed_at, "last_access_at": record.last_access_at}
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, record.path.with_suffix(".json"))

    def _new_file(self):
        reserved = sum(r.accounted - r.size for r in self.records.values())
        if shutil.disk_usage(self.directory).free < reserved + self.limits.output_bytes:
            raise OSError("缓存磁盘剩余空间不足")
        descriptor, filename = tempfile.mkstemp(prefix="tts-", suffix=".pcm", dir=self.directory)
        os.close(descriptor)
        return Path(filename)

    @staticmethod
    def _append(path, chunk):
        with path.open("ab", buffering=0) as file:
            if file.write(chunk) != len(chunk):
                raise OSError("缓存磁盘写入不完整")

    @staticmethod
    def _read(path, offset, size):
        with path.open("rb", buffering=0) as file:
            file.seek(offset)
            chunk = file.read(size)
        if len(chunk) != size:
            raise OSError("缓存磁盘读取不完整")
        return chunk

    @staticmethod
    def _sync(path):
        with path.open("rb") as file:
            os.fsync(file.fileno())

    def _drop(self, record: Record):
        try:
            self._delete_files(record)
        except OSError:
            logger.exception("删除缓存文件失败 path=%s", record.path)
            return
        if self.records.get(record.key) is record:
            del self.records[record.key]
        self.retired.discard(record)
        self.accounted_bytes -= record.accounted
        record.accounted = 0

    def _clean(self):
        now = self.clock()
        for record in list(self.records.values()) + list(self.retired):
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
                    raise CapacityError("服务繁忙，磁盘音频缓存容量不足")
                try:
                    path = self._new_file()
                except OSError as exc:
                    raise CapacityError("缓存磁盘不可写或剩余空间不足") from exc
                now = self.clock()
                record = Record(key, req, now, now, path, accounted=self.limits.output_bytes)
                self.records[key] = record
                self.accounted_bytes += record.accounted
                record.producer = asyncio.create_task(self._run(record))
                self.producers.add(record.producer)
                record.producer.add_done_callback(self.producers.discard)
            record.last_access_at = self.clock()
            if record.done:
                try:
                    await self._io(self._persist, record)
                except OSError as exc:
                    raise CapacityError("无法更新磁盘缓存，请检查磁盘空间和权限") from exc
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
            logger.info("生成开始 key=%s queued_ms=%.1f", record.key, (time.monotonic() - record.queued_at) * 1000)
            iterator = self.produce(record.request, record.key)
            try:
                async for chunk in iterator:
                    if not isinstance(chunk, bytes) or not chunk or len(chunk) % 2:
                        raise GenerationError("推理服务返回无效 PCM")
                    async with self.condition:
                        if record.size + len(chunk) > self.limits.output_bytes:
                            raise GenerationError("生成音频超过单任务输出上限")
                        await self._io(self._append, record.path, chunk)
                        if not record.size:
                            record.first_chunk_at = self.clock()
                            logger.info("共享任务收到首块 key=%s elapsed_ms=%.1f", record.key,
                                        (time.monotonic() - record.queued_at) * 1000)
                        record.size += len(chunk)
                        self.condition.notify_all()
            finally:
                await iterator.aclose()
            if not record.size:
                raise GenerationError("推理服务没有生成音频")

    async def _run(self, record: Record):
        error = None
        try:
            remaining = self.limits.task_seconds - (time.monotonic() - record.queued_at)
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(self._generate(record), remaining)
            await self._io(self._sync, record.path)
        except asyncio.CancelledError:
            error = "服务关闭，生成已中止"
        except asyncio.TimeoutError:
            error = "任务排队或生成超过总耗时上限"
        except Exception:
            logger.exception("生成失败 key=%s", record.key)
            error = "语音生成失败，请重试"
        finally:
            async with self.condition:
                record.completed_at = self.clock()
                if error is None:
                    try:
                        await self._io(self._persist, record)
                    except asyncio.CancelledError:
                        error = "服务关闭，缓存保存已中止"
                    except Exception:
                        logger.exception("保存缓存元数据失败 key=%s", record.key)
                        error = "磁盘音频缓存保存失败，请重试"
                record.done, record.error = True, error
                record.request = None
                try:
                    disk_size = record.path.stat().st_size
                except OSError:
                    disk_size = record.accounted
                self.accounted_bytes -= record.accounted - disk_size
                record.accounted = disk_size
                if error:
                    try:
                        record.path.with_suffix(".json").unlink(missing_ok=True)
                    except OSError:
                        logger.exception("删除失败缓存元数据失败 path=%s", record.path)
                    if self.records.get(record.key) is record:
                        del self.records[record.key]
                    self.retired.add(record)
                    if not record.active_readers:
                        self._drop(record)
                self.condition.notify_all()
            logger.info("生成结束 key=%s success=%s bytes=%d elapsed_ms=%.1f", record.key,
                        error is None, record.size, (time.monotonic() - record.queued_at) * 1000)

    async def first(self, record: Record):
        async with self.condition:
            await self.condition.wait_for(lambda: bool(record.size) or record.done)
            if record.error:
                raise GenerationError(record.error)
            if not record.size:
                raise GenerationError(record.error or "没有生成音频")

    async def read(self, record: Record):
        offset = 0
        while True:
            async with self.condition:
                await self.condition.wait_for(lambda: offset < record.size or record.done)
                if offset < record.size:
                    size = min(MAX_FRAME_BYTES, record.size - offset)
                elif record.error:
                    raise GenerationError(record.error)
                else:
                    return
            try:
                chunk = await self._io(self._read, record.path, offset, size)
            except OSError as exc:
                async with self.condition:
                    if record.done:
                        record.error = "磁盘音频缓存读取失败"
                        if self.records.get(record.key) is record:
                            del self.records[record.key]
                        self.retired.add(record)
                    elif record.producer:
                        record.producer.cancel()
                    self.condition.notify_all()
                raise GenerationError("磁盘音频缓存读取失败，请检查缓存目录") from exc
            offset += len(chunk)
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
        async with self.condition:
            for record in list(self.records.values()):
                if not record.done:
                    record.done, record.error, record.completed_at = True, "服务关闭，生成已中止", self.clock()
                    del self.records[record.key]
                    self.retired.add(record)
            self._clean()
            self.condition.notify_all()
