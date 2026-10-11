from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

from .common import Limits, SpeechRequest, MAX_FRAME_BYTES
from .voices import load_voice_config, prepare_clone_voices

logger = logging.getLogger(__name__)


def resolve_model_path(model_id: str):
    path = Path(model_id).expanduser()
    if path.is_dir():
        path = path.resolve()
        if not (path / "config.json").is_file():
            raise ValueError("本地模型目录缺少 config.json，请挂载完整模型仓库")
        digest = hashlib.sha256()
        for file in sorted(path.rglob("*")):
            relative = file.relative_to(path)
            if not file.is_file() or any(part.startswith(".") for part in relative.parts):
                continue
            digest.update(relative.as_posix().encode("utf-8") + b"\0")
            digest.update(str(file.stat().st_size).encode("ascii") + b"\0")
            with file.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        return str(path), "local-sha256:" + digest.hexdigest()
    if path.is_absolute() or model_id.startswith("."):
        raise ValueError("本地模型目录不存在，请检查容器内路径和挂载配置")
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(model_id, revision=os.getenv("TTS_MODEL_REVISION", "main"))
    return model_path, Path(model_path).name


class Runtime:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.model = None
        self.info = None
        self.prompts = {}
        self.lock = threading.Lock()
        self.thread = None

    def load(self):
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        if not torch.cuda.is_available():
            raise RuntimeError("推理服务需要可用的 NVIDIA CUDA GPU")
        model_id = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        model_path, revision = resolve_model_path(model_id)
        max_seq_len = int(os.getenv("TTS_MAX_SEQ_LEN", "2048"))
        max_tokens = int(os.getenv("TTS_MAX_NEW_TOKENS", "1024"))
        chunk_size = int(os.getenv("TTS_CHUNK_SIZE", "8"))
        if not 1 <= chunk_size <= 64 or not 2 <= max_tokens < max_seq_len <= 8192:
            raise ValueError("生成 token、chunk 或序列长度配置无效")
        self.model = FasterQwen3TTS.from_pretrained(
            model_path, device="cuda", dtype=torch.bfloat16, backend="torch",
            attn_implementation="sdpa", max_seq_len=max_seq_len,
        )
        model_type = self.model.model.model.tts_model_type
        if model_type not in ("base", "custom_voice"):
            raise ValueError("本服务只支持 Base 或 CustomVoice 模型")
        reference_version = None
        reference_versions = {}
        voices_file = os.getenv("TTS_VOICES_FILE", "")
        configured_default = os.getenv("TTS_DEFAULT_VOICE", "")
        if model_type == "base":
            references = load_voice_config(voices_file) if voices_file else {
                "clone": {"ref_audio": Path(os.getenv("TTS_REF_AUDIO", "/references/voice.wav")),
                          "ref_text_file": Path(os.getenv("TTS_REF_TEXT_FILE", "/references/voice.txt"))}
            }
            self.prompts, reference_versions, default_voice = prepare_clone_voices(
                self.model.model, references, configured_default,
            )
            reference_version = hashlib.sha256(
                json.dumps(reference_versions, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            voices = list(self.prompts)
        else:
            if voices_file:
                raise ValueError("自定义参考音色需要 Base 模型，CustomVoice 只支持模型内置音色")
            voices = self.model.model.get_supported_speakers() or []
            default_voice = configured_default or "vivian"
            if default_voice not in voices:
                raise ValueError("默认音色不在模型支持的音色列表中")
        languages = self.model.model.get_supported_languages() or []
        if "auto" not in [v.lower() for v in languages]:
            languages.append("Auto")
        self.model.warmup()
        torch.cuda.synchronize()
        self.info = {
            "backend": "torch-cuda-graph", "package_version": "0.5.4",
            "source_version": source_version(), "model": model_id, "model_revision": revision,
            "model_type": model_type, "reference_version": reference_version,
            "reference_versions": reference_versions,
            "sample_rate": self.model.sample_rate, "channels": 1, "sample_format": "s16le",
            "voices": voices, "default_voice": default_voice, "languages": languages,
            "supports_instruct": model_type == "base" or self.model.model.model.tts_model_size != "0b6",
            "chunk_size": chunk_size, "max_new_tokens": max_tokens, "max_seq_len": max_seq_len,
            "max_output_bytes": self.limits.output_bytes, "task_timeout_seconds": self.limits.task_seconds,
        }
        logger.info("模型加载和预热完成 info=%s allocated_bytes=%d reserved_bytes=%d",
                    self.info, torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

    def generate(self, req: SpeechRequest):
        kwargs = dict(text=req.input, language=req.language, instruct=req.instruct or None,
                      chunk_size=self.info["chunk_size"], max_new_tokens=self.info["max_new_tokens"],
                      require_eos=True)
        if self.info["model_type"] == "base":
            reference = self.prompts[req.voice]
            prompt = dict(reference["prompt"])
            for field in ("ref_spk_embedding", "ref_code"):
                prompt[field] = [value.to(self.model.device) if value is not None else None for value in prompt[field]]
            return self.model.generate_voice_clone_streaming(
                **kwargs, voice_clone_prompt=prompt, ref_text=reference["ref_text"], non_streaming_mode=False,
            )
        return self.model.generate_custom_voice_streaming(**kwargs, speaker=req.voice)

    async def stream(self, req: SpeechRequest, _key: str):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("前一个推理尚未退出，请稍后重试")
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=2)
        stopped = threading.Event()

        def emit(item):
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while True:
                try:
                    future.result(timeout=0.25)
                    return
                except concurrent.futures.TimeoutError:
                    if stopped.is_set():
                        future.cancel()
                        raise InterruptedError("共享任务已中止")

        def fatal_timeout():
            # CUDA work cannot be safely cancelled in a Python thread; restart its owner.
            logger.error("推理超过硬超时，退出服务以释放 GPU 并由 Docker 重启")
            os._exit(1)

        def produce():
            import numpy as np
            import torch
            watchdog = threading.Timer(self.limits.task_seconds, fatal_timeout)
            watchdog.daemon = True
            watchdog.start()
            start = time.monotonic()
            total = 0
            generator = None
            outcome = None
            try:
                torch.cuda.reset_peak_memory_stats()
                generator = self.generate(req)
                for audio, sr, timing in generator:
                    if stopped.is_set():
                        break
                    if sr != self.info["sample_rate"]:
                        raise ValueError("生成采样率发生变化")
                    samples = np.asarray(audio).reshape(-1)
                    if not np.isfinite(samples).all():
                        raise ValueError("生成音频包含非有限采样")
                    raw = np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()
                    if not raw:
                        continue
                    if total + len(raw) > self.limits.output_bytes:
                        raise ValueError("生成音频超过输出上限")
                    if total == 0:
                        logger.info("推理产出首块 key=%s elapsed_ms=%.1f timing=%s", _key,
                                    (time.monotonic() - start) * 1000, timing)
                    total += len(raw)
                    for offset in range(0, len(raw), MAX_FRAME_BYTES):
                        emit(raw[offset:offset + MAX_FRAME_BYTES])
                if not stopped.is_set():
                    if not total:
                        raise ValueError("模型没有生成音频")
            except InterruptedError:
                pass
            except Exception as exc:
                logger.exception("推理失败 key=%s", _key)
                outcome = exc
            finally:
                try:
                    if generator is not None:
                        generator.close()
                except Exception as exc:
                    logger.exception("推理清理失败 key=%s", _key)
                    outcome = exc
                finally:
                    watchdog.cancel()
                    self.lock.release()
                try:
                    logger.info("推理结束 key=%s bytes=%d elapsed_ms=%.1f peak_allocated_bytes=%d peak_reserved_bytes=%d",
                                _key, total, (time.monotonic() - start) * 1000,
                                torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
                except Exception:
                    logger.exception("无法读取 GPU 显存统计 key=%s", _key)
                # Signal completion only after releasing GPU ownership for the next task.
                if not stopped.is_set():
                    try:
                        emit(outcome)
                    except InterruptedError:
                        pass

        self.thread = threading.Thread(target=produce, daemon=True, name="tts-inference")
        try:
            self.thread.start()
        except BaseException:
            self.lock.release()
            raise
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stopped.set()


def source_version():
    # Include local inference changes in cache identity even before a release is committed.
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent
    for directory in (root / "faster_qwen3_tts", root / "streaming_service"):
        for path in sorted(directory.glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()
