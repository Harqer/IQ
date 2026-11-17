from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.patch_iq_native_mimo_tail import (
    KERNEL_RELATIVE_PATH,
    UPSTREAM_V_READ,
    SAFE_V_READ,
    NativeMIMOKernelSafetyError,
    kernel_tail_guard_present,
    patch_kernel,
    verify_kernel_guard,
)


class NativeMIMOTailPatchTests(unittest.TestCase):
    def test_pinned_read_rewritten_once_with_masked_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / KERNEL_RELATIVE_PATH
            path.parent.mkdir(parents=True)
            original = "before\n" + UPSTREAM_V_READ + "\nafter\n"
            path.write_text(original, encoding="utf-8")
            with self.assertRaises(NativeMIMOKernelSafetyError):
                verify_kernel_guard(path)
            patched = patch_kernel(root)
            self.assertEqual(patched, path)
            self.assertEqual(path.read_text(encoding="utf-8"),
                             original.replace(UPSTREAM_V_READ, SAFE_V_READ))
            self.assertTrue(kernel_tail_guard_present(path.read_text()))
            verify_kernel_guard(path)
            with self.assertRaisesRegex(NativeMIMOKernelSafetyError, "unrecognized"):
                patch_kernel(root)

    def test_guard_rejects_unrecognized_partial_implementation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / KERNEL_RELATIVE_PATH
            path.parent.mkdir(parents=True)
            path.write_text("other tilelang source", encoding="utf-8")
            with self.assertRaises(NativeMIMOKernelSafetyError):
                patch_kernel(root)
            with self.assertRaises(NativeMIMOKernelSafetyError):
                verify_kernel_guard(path)

    def test_guard_rejects_missing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(NativeMIMOKernelSafetyError, "missing"):
                verify_kernel_guard(Path(temp) / "missing.py")


if __name__ == "__main__":
    unittest.main()
