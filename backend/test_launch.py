import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import launch


class PreparationTests(unittest.TestCase):
    def test_references_and_case_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "voice.wav").write_bytes(b"test reference identity")
            (root / "voice.txt").write_text("这是参考文本。", encoding="utf-8")
            entry = {"ref_audio": "voice.wav", "ref_text_file": "voice.txt"}
            config = root / "voices.json"
            config.write_text(json.dumps({"Zhangsan": entry}))
            refs = launch.load_references(config)
            self.assertEqual(refs["Zhangsan"]["text"], "这是参考文本。")
            self.assertEqual(refs["Zhangsan"]["audio"], root / "voice.wav")
            config.write_text(json.dumps({"Zhangsan": entry, "zhangsan": entry}))
            with self.assertRaises(ValueError):
                launch.load_references(config)
            config.write_text('{"same": {}, "same": {}}')
            with self.assertRaises(ValueError):
                launch.load_references(config)

    def test_content_identity_and_atomic_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"first")
            before = launch.digest_tree(model)
            (model / "weights.bin").write_bytes(b"other")
            self.assertNotEqual(before, launch.digest_tree(model))
            launch.atomic_json(root / "backend.json", {"identity": before})
            self.assertEqual(json.loads((root / "backend.json").read_text())["identity"], before)
            self.assertFalse((root / "backend.tmp").exists())

    def test_config_limits(self):
        with patch.dict("os.environ", {"LIMIT": "0"}):
            with self.assertRaises(ValueError):
                launch.integer("LIMIT", 4, 1, 256)
