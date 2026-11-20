"""Read-only real HF IQ235 metadata check; NEVER downloads model weights."""
from __future__ import annotations
import json
from pathlib import Path
from iq_cloud.checkpoint_metadata import inspect_bucket

def main():
    root = Path(__file__).resolve().parents[1] / "configs"
    original = json.loads((root / "iq235_checkpoint_source.json").read_text())
    glm = json.loads((root / "iq235_glm53_tokenizer_phase_a.json").read_text())
    print(json.dumps(inspect_bucket(original, glm), indent=2))

if __name__ == "__main__":
    main()
