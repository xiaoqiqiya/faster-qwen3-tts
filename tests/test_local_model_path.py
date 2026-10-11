"""Local model resolution checks; no Torch, model loading, or network calls."""
import tempfile
import unittest
from pathlib import Path

from streaming_service.inference import resolve_model_path


class LocalModelPathTests(unittest.TestCase):
    def test_flat_hfd_directory_and_content_version(self):
        with tempfile.TemporaryDirectory(prefix=".model-path-test-", dir=Path.cwd()) as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}", encoding="utf-8")
            weights = path / "model.safetensors"
            weights.write_bytes(b"first")
            tokenizer = path / "speech_tokenizer"
            tokenizer.mkdir()
            (tokenizer / "config.json").write_text("{}", encoding="utf-8")
            model_path, revision = resolve_model_path(directory)
            self.assertEqual(model_path, str(path.resolve()))
            self.assertTrue(revision.startswith("local-sha256:"))
            self.assertEqual(resolve_model_path(directory)[1], revision)

            state = path / ".hfd"
            state.mkdir()
            (state / "status").write_text("complete", encoding="utf-8")
            self.assertEqual(resolve_model_path(directory)[1], revision)
            weights.write_bytes(b"other")
            self.assertNotEqual(resolve_model_path(directory)[1], revision)

    def test_missing_directory_or_config_is_reported_locally(self):
        with tempfile.TemporaryDirectory(prefix=".model-path-test-", dir=Path.cwd()) as directory:
            with self.assertRaisesRegex(ValueError, "config.json"):
                resolve_model_path(directory)
            with self.assertRaisesRegex(ValueError, "本地模型目录不存在"):
                resolve_model_path(str(Path(directory) / "missing"))


if __name__ == "__main__":
    unittest.main()
