"""CPU-only behavior checks for remote validation; no model or GPU needed."""
import asyncio
import io
import unittest

from examples.http_streaming_client import receive_frames
from streaming_service.cache import CapacityError, GenerationError, SharedCache, request_key
from streaming_service.common import (AUDIO, DONE, ERROR, AudioResponse, FrameDecoder, Limits,
                                      SpeechRequest, frame, resolve_request)


class ProtocolTests(unittest.TestCase):
    def test_fragmented_headers_and_samples(self):
        wire = frame(AUDIO, b"\x01\x02\x03\x04") + frame(DONE)
        decoder = FrameDecoder()
        received = []
        for byte in wire:
            received.extend(decoder.feed(bytes([byte])))
        decoder.finish()
        self.assertEqual(received, [(AUDIO, b"\x01\x02\x03\x04"), (DONE, b"")])
        self.assertEqual(list(receive_frames(io.BytesIO(wire))), [b"\x01\x02\x03\x04"])

    def test_missing_done_error_and_trailing_bytes(self):
        decoder = FrameDecoder()
        list(decoder.feed(frame(AUDIO, b"ab")))
        with self.assertRaises(ValueError):
            decoder.finish()
        with self.assertRaises(RuntimeError):
            list(receive_frames(io.BytesIO(frame(AUDIO, b"ab"))))
        with self.assertRaises(RuntimeError):
            list(receive_frames(io.BytesIO(frame(ERROR, '{"message":"失败"}'.encode()))))
        with self.assertRaises(ValueError):
            list(FrameDecoder().feed(frame(DONE) + b"x"))

    def test_keys_and_effective_defaults(self):
        info = {"default_voice": "vivian", "voices": ["vivian"], "languages": ["Chinese"],
                "supports_instruct": False, "model_revision": "one"}
        req = resolve_request(SpeechRequest(input="船名 123。"), info)
        explicit = resolve_request(SpeechRequest(input="船名 123。", voice="vivian", language="chinese"), info)
        self.assertEqual(request_key(req, info), request_key(explicit, info))
        self.assertEqual(request_key(req, info), request_key(req.model_copy(update={"response_format": "pcm"}), info))
        for updated in ({"input": "船名123。"}, {"voice": "clone"}, {"instruct": "平静"}):
            self.assertNotEqual(request_key(req, info), request_key(req.model_copy(update=updated), info))
        self.assertNotEqual(request_key(req, info), request_key(req, {**info, "model_revision": "two"}))


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = 0.0
        self.calls = 0
        self.first_three = asyncio.Event()
        self.finish = asyncio.Event()

        async def producer(_req, _key):
            self.calls += 1
            for chunk in (b"aa", b"bb", b"cc"):
                yield chunk
            self.first_three.set()
            await self.finish.wait()
            yield b"dd"

        self.limits = Limits(ttl=300, cache_bytes=32, output_bytes=8, pending=2,
                             readers=256, task_seconds=2, send_seconds=0.02)
        self.cache = SharedCache(self.limits, producer, clock=lambda: self.now)
        self.req = SpeechRequest(input="船舶播报")

    async def asyncTearDown(self):
        await self.cache.close()

    async def collect(self, record):
        try:
            return [chunk async for chunk in self.cache.read(record)]
        finally:
            await self.cache.release(record)

    async def test_100_same_requests_and_late_reader(self):
        records = await asyncio.gather(*(self.cache.acquire("same", self.req) for _ in range(100)))
        await asyncio.wait_for(self.first_three.wait(), 1)
        self.assertEqual(self.calls, 1)
        self.assertTrue(all(record is records[0] for record in records))
        self.assertFalse(records[0].done)
        late = await self.cache.acquire("same", self.req)
        reader = self.cache.read(late)
        history = [await anext(reader) for _ in range(3)]
        self.assertEqual(history, [b"aa", b"bb", b"cc"])
        self.finish.set()
        history.extend([chunk async for chunk in reader])
        await self.cache.release(late)
        results = await asyncio.gather(*(self.collect(record) for record in records))
        self.assertEqual(history, [b"aa", b"bb", b"cc", b"dd"])
        self.assertTrue(all(result == history for result in results))

    async def test_disconnect_and_slow_reader_do_not_hold_producer(self):
        disconnected = await self.cache.acquire("same", self.req)
        slow = await self.cache.acquire("same", self.req)
        await self.cache.release(disconnected)
        await asyncio.wait_for(self.first_three.wait(), 1)
        self.finish.set()
        await slow.producer
        self.assertTrue(slow.done)
        self.assertEqual(await self.collect(slow), [b"aa", b"bb", b"cc", b"dd"])
        self.assertEqual(self.calls, 1)

    async def test_backend_unavailable_only_allows_existing_cache(self):
        self.finish.set()
        record = await self.cache.acquire("same", self.req)
        await self.collect(record)
        hit = await self.cache.acquire("same", self.req, allow_new=False)
        self.assertIs(hit, record)
        self.assertEqual(await self.collect(hit), [b"aa", b"bb", b"cc", b"dd"])
        with self.assertRaises(CapacityError):
            await self.cache.acquire("uncached", self.req, allow_new=False)
        self.assertEqual(self.calls, 1)

    async def test_sliding_ttl_and_active_reader_protection(self):
        record = await self.cache.acquire("same", self.req)
        await asyncio.wait_for(self.first_three.wait(), 1)
        self.now = 50
        self.finish.set()
        await record.producer
        await self.collect(record)
        self.now = 349
        hit = await self.cache.acquire("same", self.req)
        self.assertIs(hit, record)
        await self.collect(hit)
        self.now = 648
        hit = await self.cache.acquire("same", self.req)
        self.now = 1000
        async with self.cache.condition:
            self.cache._clean()
        self.assertIn("same", self.cache.records)
        await self.collect(hit)
        self.assertNotIn("same", self.cache.records)
        retry = await self.cache.acquire("same", self.req)
        await self.collect(retry)
        self.assertEqual(self.calls, 2)

    async def test_failure_detaches_partial_result_and_retries(self):
        allow_failure = asyncio.Event()

        async def failing(_req, _key):
            yield b"aa"
            await allow_failure.wait()
            raise RuntimeError("mock backend failure")

        self.cache.produce = failing
        record = await self.cache.acquire("same", self.req)
        await self.cache.first(record)
        allow_failure.set()
        await record.producer
        self.assertNotIn("same", self.cache.records)
        self.assertEqual(self.cache.accounted_bytes, 2)
        reader = self.cache.read(record)
        self.assertEqual(await anext(reader), b"aa")
        with self.assertRaises(GenerationError):
            await anext(reader)

        async def good(_req, _key):
            yield b"bb"

        self.cache.produce = good
        retry = await self.cache.acquire("same", self.req)
        self.assertIsNot(retry, record)
        await self.cache.release(record)
        self.assertIs(self.cache.records["same"], retry)
        self.assertEqual(await self.collect(retry), [b"bb"])
        self.assertEqual(self.cache.accounted_bytes, 2)

    async def test_lru_reservations_and_active_records(self):
        self.finish.set()
        self.cache.limits = Limits(cache_bytes=16, output_bytes=8)
        first = await self.cache.acquire("first", self.req)
        await self.collect(first)
        self.now = 1
        pinned = await self.cache.acquire("pinned", self.req)
        await pinned.producer
        self.now = 2
        third = await self.cache.acquire("third", self.req)
        self.assertNotIn("first", self.cache.records)
        self.assertIn("pinned", self.cache.records)
        self.assertLessEqual(self.cache.accounted_bytes, 16)
        with self.assertRaises(CapacityError):
            await self.cache.acquire("fourth", self.req)
        await self.collect(pinned)
        await self.collect(third)

    async def test_queue_limit_output_limit_and_deadline(self):
        self.cache.limits = Limits(cache_bytes=32, output_bytes=8, pending=1, task_seconds=0.05)
        one = await self.cache.acquire("one", self.req)
        two = await self.cache.acquire("two", self.req)
        with self.assertRaises(CapacityError):
            await self.cache.acquire("three", self.req)
        for record in (one, two):
            with self.assertRaises(GenerationError):
                await self.collect(record)

        async def oversized(_req, _key):
            yield b"a" * 10

        self.cache.produce = oversized
        record = await self.cache.acquire("oversized", self.req)
        with self.assertRaises(GenerationError):
            await self.collect(record)
        self.assertEqual(self.cache.accounted_bytes, 0)

    async def test_send_timeout_releases_reader_even_before_body(self):
        released = asyncio.Event()

        async def body():
            yield b"aa"

        async def release():
            released.set()

        async def stalled_send(_message):
            await asyncio.Event().wait()

        async def receive():
            await asyncio.Event().wait()

        response = AudioResponse(body(), sample_rate=24000, fmt="pcm_framed", send_seconds=0.01,
                                 release=release)
        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
        with self.assertRaises(asyncio.TimeoutError):
            await response(scope, receive, stalled_send)
        self.assertTrue(released.is_set())


if __name__ == "__main__":
    unittest.main()
