"""Emit real-bucket IQ235->GLM migration inventory; JSON metadata ONLY."""
from __future__ import annotations

import json
from pathlib import Path

from iq_cloud.checkpoint_migration import inspect_remote_migration_plan


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "configs"
    original = json.loads((root / "iq235_checkpoint_source.json").read_text())
    target = json.loads((root / "iq235_glm53_tokenizer_phase_a.json").read_text())
    print(json.dumps(inspect_remote_migration_plan(original, target),
                     indent=2, sort_keys=True))
