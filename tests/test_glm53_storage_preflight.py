import unittest
from scripts.full78_storage_preflight import (
    EXPERT_SLOT_COUNT, REAL_ROUTED_SLOT_COUNT, required_private_bytes,
    verify_capacity,
)


class Full78StoragePreflight(unittest.TestCase):
    def test_measured_checkpoint_capacity_above_free_quota(self):
        self.assertEqual(REAL_ROUTED_SLOT_COUNT, 19200)
        self.assertEqual(EXPERT_SLOT_COUNT, 19968)
        self.assertGreater(required_private_bytes(), 600_000_000_000)
        self.assertLess(required_private_bytes(), 700_000_000_000)

    def test_free_private_quota_fails_before_spending(self):
        with self.assertRaisesRegex(RuntimeError, "FULL78_STORAGE_BLOCKED"):
            verify_capacity(100_000_000_000)

    def test_adequate_private_capacity_passes(self):
        verify_capacity(900_000_000_000)


if __name__ == "__main__":
    unittest.main()
