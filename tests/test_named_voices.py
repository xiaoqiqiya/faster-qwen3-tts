"""Named clone configuration and prompt selection, without a GPU or model load."""
import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI, HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException

from streaming_service.cache import request_key
from streaming_service.common import Limits, SpeechRequest, install_errors, resolve_request
from streaming_service.inference import Runtime
from streaming_service.voices import load_voice_config, prepare_clone_voices


class FakeTensor:
    def __init__(self, device="cuda"):
        self.device = device

    def detach(self):
        return self

    def cpu(self):
        return FakeTensor("cpu")

    def to(self, device):
        return FakeTensor(device)


class NamedVoiceTests(unittest.TestCase):
    def test_names_and_paths_are_loaded_without_aliases(self):
        with tempfile.TemporaryDirectory(prefix=".voices-test-", dir=Path.cwd()) as directory:
            path = Path(directory)
            absolute = str(path / "lisi.wav")
            config = path / "voices.json"
            config.write_text(json.dumps({
                "zhangsan": {"ref_audio": "zhangsan.wav", "ref_text_file": "zhangsan.txt"},
                "lisi": {"ref_audio": absolute, "ref_text_file": "lisi.txt"},
            }), encoding="utf-8")
            loaded = load_voice_config(str(config))
            self.assertEqual(list(loaded), ["zhangsan", "lisi"])
            self.assertEqual(loaded["zhangsan"]["ref_audio"], path / "zhangsan.wav")
            self.assertEqual(loaded["lisi"]["ref_audio"], Path(absolute))

    def test_duplicate_names_and_invalid_entries_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".voices-test-", dir=Path.cwd()) as directory:
            path = Path(directory) / "voices.json"
            path.write_text('{"clone": {}, "clone": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复"):
                load_voice_config(str(path))
            for invalid in ({}, [], {" ": {}}, {"someone": {"ref_audio": "a.wav"}},
                            {"someone": {"ref_audio": "", "ref_text_file": "a.txt"}}):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_voice_config(str(path))

    def test_precompute_once_per_voice_then_select_by_request_name(self):
        with tempfile.TemporaryDirectory(prefix=".voices-test-", dir=Path.cwd()) as directory:
            path = Path(directory)
            references = {}
            for name in ("zhangsan", "lisi"):
                audio, text = path / f"{name}.wav", path / f"{name}.txt"
                audio.write_bytes(name.encode())
                text.write_text(f"{name} 的参考转写", encoding="utf-8")
                references[name] = {"ref_audio": audio, "ref_text_file": text}
            samples = types.SimpleNamespace(mean=lambda axis: [0.1, 0.2])
            fake_numpy = types.SimpleNamespace(
                isfinite=lambda audio: types.SimpleNamespace(all=lambda: True),
                zeros=lambda count, dtype: [0.0] * count,
                concatenate=lambda arrays: arrays[0] + arrays[1],
            )
            fake_soundfile = types.SimpleNamespace(
                info=lambda path: types.SimpleNamespace(duration=2),
                read=lambda path, **kwargs: (samples, 10),
            )
            model = types.SimpleNamespace(create_voice_clone_prompt=Mock(side_effect=lambda **kw: [
                types.SimpleNamespace(ref_spk_embedding=FakeTensor(), ref_code=FakeTensor(),
                                      x_vector_only_mode=False, icl_mode=True)
            ]))
            with patch.dict(sys.modules, {"numpy": fake_numpy, "soundfile": fake_soundfile}):
                prompts, versions, default = prepare_clone_voices(model, references, "lisi")
                self.assertEqual(model.create_voice_clone_prompt.call_count, 2)
                self.assertEqual(default, "lisi")
                self.assertNotEqual(versions["zhangsan"], versions["lisi"])
                self.assertEqual(prompts["zhangsan"]["ref_text"], "zhangsan 的参考转写")
                self.assertEqual(len(model.create_voice_clone_prompt.call_args_list[0].kwargs["ref_audio"][0]), 7)
                for name in prompts:
                    for field in ("ref_spk_embedding", "ref_code"):
                        self.assertEqual(prompts[name]["prompt"][field][0].device, "cpu")
                with self.assertRaisesRegex(ValueError, "默认音色"):
                    prepare_clone_voices(model, references, "unknown")

            runtime = Runtime(Limits())
            runtime.prompts = prompts
            runtime.info = {"model_type": "base", "chunk_size": 8, "max_new_tokens": 1024}
            runtime.model = types.SimpleNamespace(device="cuda", generate_voice_clone_streaming=Mock(return_value=iter(())))
            runtime.generate(SpeechRequest(input="播报", voice="lisi"))
            call = runtime.model.generate_voice_clone_streaming.call_args.kwargs
            self.assertEqual(call["ref_text"], "lisi 的参考转写")
            self.assertEqual(call["voice_clone_prompt"]["ref_code"][0].device, "cuda")
            self.assertEqual(call["voice_clone_prompt"]["ref_spk_embedding"][0].device, "cuda")
            self.assertEqual(prompts["lisi"]["prompt"]["ref_code"][0].device, "cpu")
            self.assertEqual(prompts["zhangsan"]["prompt"]["ref_code"][0].device, "cpu")
            self.assertEqual(model.create_voice_clone_prompt.call_count, 2)

    def test_default_selection_list_and_cache_isolation(self):
        info = {"voices": ["zhangsan", "lisi"], "default_voice": "zhangsan", "languages": ["Chinese"],
                "supports_instruct": True, "reference_versions": {"zhangsan": "first", "lisi": "second"}}
        default = resolve_request(SpeechRequest(input="同一条播报"), info)
        other = resolve_request(SpeechRequest(input="同一条播报", voice="lisi"), info)
        self.assertEqual(default.voice, "zhangsan")
        self.assertNotEqual(request_key(default, info), request_key(other, info))
        with self.assertRaises(HTTPException) as raised:
            resolve_request(SpeechRequest(input="播报", voice="clone"), info)
        self.assertEqual(raised.exception.status_code, 404)

    def test_unknown_voice_message_uses_uniform_404_response(self):
        app = FastAPI()
        install_errors(app)
        handler = app.exception_handlers[StarletteHTTPException]
        response = asyncio.run(handler(None, HTTPException(404, "指定音色不存在")))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(json.loads(response.body), {"code": 404, "message": "指定音色不存在", "data": None})


if __name__ == "__main__":
    unittest.main()
