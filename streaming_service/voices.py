"""Named clone references loaded once when the inference service starts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"音色配置存在重复名称或字段：{key}")
        result[key] = value
    return result


def load_voice_config(config_file: str):
    path = Path(config_file)
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    except OSError as exc:
        raise ValueError("无法读取音色配置文件，请检查挂载和文件路径") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("音色配置文件不是有效 JSON") from exc
    if not isinstance(config, dict) or not 1 <= len(config) <= 32:
        raise ValueError("音色配置必须包含 1 到 32 个命名音色")
    result = {}
    for name, entry in config.items():
        if not name.strip() or len(name) > 128:
            raise ValueError("音色名称不能为空或超过 128 字符")
        if not isinstance(entry, dict) or set(entry) != {"ref_audio", "ref_text_file"}:
            raise ValueError(f"音色 {name} 必须配置 ref_audio 和 ref_text_file")
        paths = {}
        for key, value in entry.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"音色 {name} 的 {key} 必须是非空文件路径")
            reference = Path(value)
            paths[key] = reference if reference.is_absolute() else path.parent / reference
        result[name] = paths
    return result


def prepare_clone_voices(model, references: dict, default_voice: str = ""):
    import numpy as np
    import soundfile as sf

    default_voice = default_voice or next(iter(references))
    if default_voice not in references:
        raise ValueError("默认音色不在已配置的克隆音色列表中")
    prompts, versions = {}, {}
    for name, files in references.items():
        audio_path, text_path = files["ref_audio"], files["ref_text_file"]
        if not audio_path.is_file() or not text_path.is_file():
            raise ValueError(f"音色 {name} 的参考录音或转写文件不存在")
        text = text_path.read_text(encoding="utf-8-sig").strip()
        if not text or len(text) > 2048:
            raise ValueError(f"音色 {name} 的参考转写不能为空或超过 2048 字符")
        audio_info = sf.info(str(audio_path))
        if not 0 < audio_info.duration <= 30:
            raise ValueError(f"音色 {name} 的参考录音时长必须在 0 到 30 秒之间")
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if not np.isfinite(audio).all():
            raise ValueError(f"音色 {name} 的参考录音包含非有限采样")
        audio = np.concatenate([audio.mean(axis=1), np.zeros(int(sr * 0.5), dtype=np.float32)])
        items = model.create_voice_clone_prompt(
            ref_audio=(audio, sr), ref_text=text, x_vector_only_mode=False,
        )
        if len(items) != 1:
            raise ValueError(f"音色 {name} 未生成有效的单音色参考特征")
        # Keep all named references on CPU; only the selected voice enters the GPU.
        prompts[name] = {
            "ref_text": text,
            "prompt": {
                "ref_spk_embedding": [item.ref_spk_embedding.detach().cpu() for item in items],
                "ref_code": [item.ref_code.detach().cpu() if item.ref_code is not None else None for item in items],
                "x_vector_only_mode": [bool(item.x_vector_only_mode) for item in items],
                "icl_mode": [bool(item.icl_mode) for item in items],
            },
        }
        versions[name] = hashlib.sha256(audio_path.read_bytes() + text.encode("utf-8")).hexdigest()
    return prompts, versions, default_voice
