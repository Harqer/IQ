"""CPU-only tests for bounded IQ H200 shard staging. No paid Jobs."""
from __future__ import annotations

import tempfile
import errno
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.iq_h200_staging import iter_staged_shards


class StagingTests(unittest.TestCase):
    def test_direct_bucket_samples_bounded_and_cleanup(self):
        from scripts.benchmark_iq_h200_staging import iter_bucket_samples, BUCKET
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = [f"model-context-{i:03d}.safetensors" for i in range(4)]
            def fake_metadata(bucket, remote):
                self.assertEqual(bucket, BUCKET)
                self.assertTrue(remote.startswith("IQ/runs/"))
                return SimpleNamespace(size=16)

            def fake_download(bucket, *, files):
                self.assertEqual(bucket, BUCKET)
                self.assertEqual(len(files), 1)
                _, local = files[0]
                Path(local).write_bytes(b"4" * 16)

            with patch("huggingface_hub.get_bucket_file_metadata",
                       side_effect=fake_metadata):
                with patch("huggingface_hub.download_bucket_files",
                           side_effect=fake_download):
                    completed = []
                    for name, path, size, _ in iter_bucket_samples(BUCKET, names, 2):
                        self.assertEqual(path.read_bytes(), b"4" * 16)
                        self.assertEqual(size, 16)
                        completed.append(name)
                    self.assertEqual(completed, names)

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


    def test_bucket_eio_resumes_at_verified_last_complete_byte(self):
        from scripts.iq_h200_staging import _copy_shard
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "remote.safetensors"
            target = root / "staged.safetensors"
            original_bytes = bytes(range(256)) * 40
            source.write_bytes(original_bytes)
            original_open = Path.open
            observed_seeks = []
            injected = {"error": False}

            class IntermittentSource:
                def __init__(self, file):
                    self.file = file
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    self.file.close()
                def seek(self, offset):
                    observed_seeks.append(offset)
                    return self.file.seek(offset)
                def read(self, count):
                    if self.file.tell() >= 128 and not injected["error"]:
                        injected["error"] = True
                        raise OSError(errno.EIO, "Simulated transient remote FUSE EIO")
                    return self.file.read(count)
                @property
                def closed(self):
                    return self.file.closed
                def close(self):
                    self.file.close()

            def patched_open(self, mode="r", *args, **kwargs):
                file = original_open(self, mode, *args, **kwargs)
                if self == source and mode == "rb":
                    return IntermittentSource(file)
                return file

            with patch.object(Path, "open", patched_open), \
                 patch("scripts.iq_h200_staging.COPY_BUFFER_BYTES", 64), \
                 patch("scripts.iq_h200_staging.sleep"):
                size, _ = _copy_shard(source, target)
            self.assertTrue(injected["error"])
            self.assertEqual(size, len(original_bytes))
            self.assertEqual(target.read_bytes(), original_bytes)
            self.assertIn(128, observed_seeks)
            self.assertFalse((root / "staged.safetensors.partial").exists())

    def test_permanent_bucket_eio_exhausts_retries_without_publishing(self):
        from scripts.iq_h200_staging import _copy_shard, MAX_READ_RETRIES
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "remote.safetensors"
            destination = root / "local.safetensors"
            source.write_bytes(b"original" * 100)
            original_open = Path.open
            failures = [0]

            def fail_open(self, mode="r", *args, **kwargs):
                if self == source and mode == "rb":
                    failures[0] += 1
                    raise OSError(errno.EIO, "Simulated persistently unavailable bucket")
                return original_open(self, mode, *args, **kwargs)

            with patch.object(Path, "open", fail_open), \
                 patch("scripts.iq_h200_staging.sleep"):
                with self.assertRaisesRegex(IOError, "Remote checkpoint read exhausted"):
                    _copy_shard(source, destination)
            self.assertEqual(failures[0], MAX_READ_RETRIES + 1)
            self.assertFalse(destination.exists())
            self.assertFalse((root / "local.safetensors.partial").exists())
            self.assertEqual(source.read_bytes(), b"original" * 100)

    def test_short_or_changed_shard_never_published(self):
        from scripts.iq_h200_staging import _copy_shard
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "remote.safetensors"
            target = root / "local.safetensors"
            source.write_bytes(b"original" * 100)
            original_open = Path.open

            class EarlyEOFSource:
                def __init__(self, file):
                    self.file = file
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    self.file.close()
                def seek(self, offset):
                    return self.file.seek(offset)
                def read(self, count):
                    return b""
                @property
                def closed(self):
                    return self.file.closed
                def close(self):
                    self.file.close()

            def short_open(self, mode="r", *args, **kwargs):
                file = original_open(self, mode, *args, **kwargs)
                if self == source and mode == "rb":
                    return EarlyEOFSource(file)
                return file

            with patch.object(Path, "open", short_open), \
                 patch("scripts.iq_h200_staging.sleep"):
                with self.assertRaisesRegex(IOError, "Remote checkpoint read exhausted"):
                    _copy_shard(source, target)
            self.assertFalse(target.exists())
            self.assertFalse((root / "local.safetensors.partial").exists())


if __name__ == "__main__":
    unittest.main()
