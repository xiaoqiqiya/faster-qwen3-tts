#!/usr/bin/env python3
"""Read framed PCM without treating a clean transport close as synthesis success."""
from __future__ import annotations

import argparse
import json
import struct
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

# Kept stdlib-only so a playback client does not need the server/model dependencies.
HEADER = struct.Struct("!BI")


def receive_frames(response):
    def read_exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            block = response.read(size - len(chunks))
            if not block:
                raise RuntimeError("音频连接中断或缺少显式完成帧")
            chunks.extend(block)
        return bytes(chunks)

    while True:
        kind, size = HEADER.unpack(read_exact(HEADER.size))
        if kind not in (1, 2, 3) or size > 262144:
            raise RuntimeError("音频帧头无效")
        if kind == 1 and (not size or size % 2):
            raise RuntimeError("PCM 音频帧采样不完整")
        if kind == 2 and size:
            raise RuntimeError("完成帧无效")
        payload = read_exact(size)
        if kind == 3:
            raise RuntimeError(json.loads(payload).get("message", "语音生成失败"))
        if kind == 2:
            if response.read(1):
                raise RuntimeError("完成帧后出现额外数据")
            return
        yield payload


def main():
    parser = argparse.ArgumentParser(description="逐块接收、保存和播放远端 TTS 音频")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--output", default="speech.wav")
    parser.add_argument("--play", action="store_true", help="使用 sounddevice 的一个持续播放器")
    parser.add_argument("--timeout", type=float, default=150)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("输出文件已存在，请指定新的文件名")
    stream = None
    if args.play:
        try:
            import sounddevice
        except ImportError:
            parser.error("播放需要安装 sounddevice：pip install sounddevice")
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/audio/speech",
        data=json.dumps({"input": args.text, "voice": args.voice, "language": args.language,
                         "instruct": args.instruct, "response_format": "pcm_framed"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.monotonic()
    created = False
    succeeded = False
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            if response.headers.get("X-Audio-Protocol") != "qwen-pcm-v1":
                raise RuntimeError("服务未返回约定的显式完成协议")
            if response.headers.get("X-Audio-Sample-Format") != "s16le" or response.headers.get("X-Audio-Channels") != "1":
                raise RuntimeError("服务音频格式不受支持")
            sr = int(response.headers["X-Audio-Sample-Rate"])
            with output.open("xb") as file, wave.open(file, "wb") as wav:
                created = True
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                if args.play:
                    stream = sounddevice.RawOutputStream(samplerate=sr, channels=1, dtype="int16")
                    stream.start()
                count = 0
                for chunk in receive_frames(response):
                    if count == 0:
                        print(f"客户端收到首块 monotonic={time.monotonic():.6f} elapsed_ms={(time.monotonic() - start) * 1000:.1f}")
                    count += 1
                    wav.writeframesraw(chunk)
                    if stream:
                        stream.write(chunk)
                if not count:
                    raise RuntimeError("服务返回了空音频")
        succeeded = True
        print(f"收到完成帧 chunks={count} elapsed_ms={(time.monotonic() - start) * 1000:.1f} output={output}")
    except urllib.error.HTTPError as exc:
        raise SystemExit(exc.read().decode("utf-8", errors="replace")) from exc
    finally:
        if stream:
            stream.stop() if succeeded else stream.abort()
            stream.close()
        if created and not succeeded:
            # Played audio cannot be recalled; never keep a partial file as a success result.
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
