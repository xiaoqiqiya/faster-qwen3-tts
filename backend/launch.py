"""Prepare immutable Qwen3-TTS metadata, then exec the pinned vLLM-Omni server."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ENGINE = "vllm-omni-v0.28.0-sha256:6f8be103eaf0055448cf7578cfd621405fd669079d4361bd58896326b2bf722a"
UPSTREAM = Path("/app/vllm-omni")
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


def integer(name: str, default: int, low: int, high: int) -> int:
    value = int(os.environ.get(name) or default)
    if not low <= value <= high:
        raise ValueError(f"{name} 必须在 {low} 到 {high} 之间")
    return value


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        relative = file.relative_to(path)
        if file.is_file() and not any(part.startswith(".") for part in relative.parts):
            digest.update(str(relative).encode())
            digest.update(bytes.fromhex(digest_file(file)))
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"配置字段重复：{key}")
        result[key] = value
    return result


def load_references(config: Path) -> dict:
    refs = json.loads(config.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
    if not isinstance(refs, dict) or not 1 <= len(refs) <= 32:
        raise ValueError("音色配置必须包含 1 到 32 个音色")
    result, lowered = {}, set()
    for name, entry in refs.items():
        if (not name.strip() or name != name.strip() or len(name) > 128
                or name in {".", ".."} or any(c in name for c in "/\\\0") or name.lower() in lowered):
            raise ValueError("音色名称无效或忽略大小写后重复")
        lowered.add(name.lower())
        if not isinstance(entry, dict) or set(entry) != {"ref_audio", "ref_text_file"}:
            raise ValueError(f"音色 {name} 必须配置 ref_audio 和 ref_text_file")
        paths = {}
        for key, value in entry.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"音色 {name} 的文件路径无效")
            path = Path(value)
            paths[key] = path if path.is_absolute() else config.parent / path
        text = paths["ref_text_file"].read_text(encoding="utf-8-sig").strip()
        if not text or len(text) > 2048:
            raise ValueError(f"音色 {name} 的转写必须为 1 到 2048 字符")
        result[name] = {"audio": paths["ref_audio"], "text": text,
                        "hash": digest_file(paths["ref_audio"])}
    return result


def prepare_profiles(model: Path, refs: dict, destination: Path) -> None:
    import numpy as np
    import soundfile as sf

    if (destination / ".complete").is_file():
        expected = json.loads((destination / ".complete").read_text())
        if expected == {p.name: digest_file(p) for p in destination.iterdir() if p.is_file() and p.name != ".complete"}:
            return
        raise ValueError("持久化音色特征损坏，请删除对应 profiles 子目录后重新启动")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="precompute-", dir=destination.parent))
    try:
        for name, ref in refs.items():
            audio_info = sf.info(str(ref["audio"]))
            if not 0 < audio_info.duration <= 30 or not 1 <= audio_info.channels <= 8 or not 8000 <= audio_info.samplerate <= 192000:
                raise ValueError(f"音色 {name} 的录音必须为 0 到 30 秒、1 到 8 声道、8 到 192 kHz")
            audio, sr = sf.read(str(ref["audio"]), dtype="float32", always_2d=True)
            if not np.isfinite(audio).all():
                raise ValueError(f"音色 {name} 包含无效采样")
            audio = np.concatenate([audio.mean(axis=1), np.zeros(int(sr * 0.5), dtype=np.float32)])
            prepared = temporary / "reference.wav"
            sf.write(str(prepared), audio, sr, subtype="FLOAT")
            subprocess.run([sys.executable, str(UPSTREAM / "examples/online_serving/text_to_speech/qwen3_tts/precompute_custom_voice.py"),
                            "--model", str(model), "--voice-name", name.lower(),
                            "--ref-audio", str(prepared), "--ref-text", ref["text"],
                            "--mode", "icl", "--output-dir", str(temporary), "--device", "cuda"], check=True)
            prepared.unlink()
        checksums = {p.name: digest_file(p) for p in temporary.iterdir() if p.is_file()}
        (temporary / ".complete").write_text(json.dumps(checksums), encoding="utf-8")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    import yaml

    state = Path(os.environ.get("TTS_STATE_DIR", "/state"))
    state.mkdir(parents=True, exist_ok=True)
    # The gateway also verifies the served model ID, so stale metadata is never trusted.
    (state / "backend.json").unlink(missing_ok=True)
    model_id = os.environ.get("TTS_MODEL") or "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    model = Path(model_id)
    if not model.is_dir():
        if model.is_absolute():
            raise ValueError(f"模型目录不存在：{model}")
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(model_id, revision=os.environ.get("TTS_MODEL_REVISION") or "main", cache_dir="/models/hub"))
    model = model.resolve()
    hf = json.loads((model / "config.json").read_text())
    if hf.get("model_type") != "qwen3_tts" or hf.get("tokenizer_type") != "qwen3_tts_tokenizer_12hz":
        raise ValueError("只支持 Qwen3-TTS 12Hz 模型")
    task = {"base": "Base", "custom_voice": "CustomVoice"}.get(hf.get("tts_model_type"))
    if task is None:
        raise ValueError("仅支持 Base 和 CustomVoice 模型")
    concurrency = integer("TTS_MAX_CONCURRENT_TASKS", 4, 1, 256)
    tokens = integer("TTS_MAX_NEW_TOKENS", 1024, 2, 4096)
    memory = float(os.environ.get("TTS_GPU_MEMORY_UTILIZATION") or "0.3")
    if not math.isfinite(memory) or not 0 < memory < 0.5:
        raise ValueError("每个推理阶段的显存比例必须大于 0 且小于 0.5")
    deploy = yaml.safe_load((UPSTREAM / "vllm_omni/deploy/qwen3_tts.yaml").read_text())
    for stage in deploy["stages"]:
        stage["max_num_seqs"] = concurrency
        stage["gpu_memory_utilization"] = memory
    deploy["stages"][0]["default_sampling_params"]["max_tokens"] = tokens
    extra = deploy["connectors"]["connector_of_shared_memory"]["extra"]
    extra["decode_batch_max_size"] = min(concurrency, 4)
    refs = {}
    if task == "Base":
        refs = load_references(Path(os.environ.get("TTS_VOICES_FILE") or "/references/voices.json"))
        voices = list(refs)
    else:
        voices = list(hf["talker_config"]["spk_id"])
    default = os.environ.get("TTS_DEFAULT_VOICE") or ("vivian" if task == "CustomVoice" else voices[0])
    if default not in voices:
        raise ValueError("默认音色不在已配置的音色列表中")
    identity_data = {"engine": ENGINE, "launcher": digest_file(Path(__file__)), "model": digest_tree(model),
                     "deploy": deploy, "refs": {n: {"hash": r["hash"], "text": r["text"]} for n, r in refs.items()}}
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    if refs:
        profiles = state / "profiles" / identity
        prepare_profiles(model, refs, profiles)
        deploy["custom_voice_dir"] = str(profiles)
        identity = hashlib.sha256((identity + digest_tree(profiles)).encode()).hexdigest()
    info = {"identity": identity, "model": str(model), "served_model": "tts-" + identity,
            "task_type": task, "voices": voices, "default_voice": default,
            "languages": LANGUAGES, "sample_rate": 24000,
            "supports_instruct": task == "CustomVoice" and hf.get("tts_model_size") == "1b7",
            "max_new_tokens": tokens, "concurrency": concurrency}
    config = state / "deploy.yaml"
    config.write_text(yaml.safe_dump(deploy), encoding="utf-8")
    atomic_json(state / "backend.json", info)
    os.execvp("vllm", ["vllm", "serve", str(model), "--omni", "--deploy-config", str(config),
                       "--served-model-name", info["served_model"], "--host", "0.0.0.0", "--port", "8001"])


if __name__ == "__main__":
    main()
