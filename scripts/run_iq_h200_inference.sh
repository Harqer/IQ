#!/usr/bin/env bash
# Compatibility entrypoint. The canonical launcher is the documented Python SDK path.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$root/scripts/iq_hf_inference.py" "$@"
