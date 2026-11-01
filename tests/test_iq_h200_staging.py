"""CPU-only tests for bounded IQ H200 shard staging. No paid Jobs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.iq_h200_staging import iter_staged_shards


class StagingTests(unittest.TestCase):
    def test_staging_preserves_bytes_uses_bounded_window_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            local = root / "local"
            source.mkdir()
            local.mkdir()
            names = [f"model-context-{i:03d}.safetensors" for i in range(12)]
            for i, name in enumerate(names):
                (source / name).write_bytes(bytes([i]) * (1024 * 64 + i))
            actual = []
            for name, path, size, seconds in iter_staged_shards(
                source, names, workers=4, stage_parent=local
            ):
                actual.append(name)
                self.assertEqual(path.read_bytes(), (source / name).read_bytes())
                self.assertEqual(size, (source / name).stat().st_size)
                self.assertGreaterEqual(seconds, 0)
                staged = list(next(local.iterdir()).iterdir())
                self.assertLessEqual(len(staged), 4)
            self.assertEqual(actual, names)
            self.assertEqual(list(local.iterdir()), [])

    def test_missing_shard_cleans_up_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            stage = root / "stage"
            source.mkdir()
            stage.mkdir()
            (source / "part-0.safetensors").write_bytes(b"data")
            with self.assertRaises(FileNotFoundError):
                list(iter_staged_shards(source,
                    ["part-0.safetensors", "part-1.safetensors"],
                    workers=2, stage_parent=stage))
            self.assertFalse(list(stage.iterdir()))
            self.assertEqual((source / "part-0.safetensors").read_bytes(), b"data")

    def test_path_symlink_and_size_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "part-0.safetensors").write_bytes(b"data")
            (source / "alias.safetensors").symlink_to("part-0.safetensors")
            for names in (["../escape.safetensors"], ["part-0.safetensors"] * 2):
                with self.assertRaises(ValueError):
                    list(iter_staged_shards(source, names))
            with self.assertRaises(FileNotFoundError):
                list(iter_staged_shards(source, ["alias.safetensors"]))
            with self.assertRaises(ValueError):
                list(iter_staged_shards(source, ["part-0.safetensors"], stage_parent=source))
            with self.assertRaises(ValueError):
                list(iter_staged_shards(source, ["part-0.safetensors"], workers=9))


if __name__ == "__main__":
    unittest.main()
