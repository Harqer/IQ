#!/usr/bin/env python3
"""Read-only GLM-first all256 preflight with OpenTelemetry, no GPU/inference.

Only opens local metadata, tokenizer mappings, safetensors headers, and (when
explicitly requested) streams local shard bytes to SHA-256. NEVER deploys.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from pathlib import Path

from audit_iq_foundation_contract import audit as audit_foundation
from iq_telemetry import IQTelemetry

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def index_inventory(root: Path, expected_shards: int | None) -> tuple[dict, list[str]]:
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weights = index.get("weight_map")
    require(isinstance(weights, dict) and bool(weights), "Missing weight_map")
    require(all(isinstance(key, str) and key for key in weights),
            "Malformed tensor names")
    require(all(isinstance(name, str) and name.endswith(".safetensors")
                and Path(name).name == name and name not in {".", ".."}
                for name in weights.values()), "Invalid safetensors filename")
    shards = sorted(set(weights.values()))
    if expected_shards is not None:
        require(len(shards) == expected_shards, "Indexed shard count differs from expected")
    for name in shards:
        path = root / name
        require(path.is_file() and path.stat().st_size > 0,
                "Missing or empty indexed safetensors shard")
    return weights, shards


def verify_headers(root: Path, weights: dict[str, str], shards: list[str]) -> int:
    """Inspect safetensors header keys, shapes, dtype; never materialize arrays."""
    from safetensors import safe_open

    verified = 0
    by_shard: dict[str, set[str]] = {name: set() for name in shards}
    for tensor_name, filename in weights.items():
        by_shard[filename].add(tensor_name)
    for filename in shards:
        with safe_open(str(root / filename), framework="np", device="cpu") as handle:
            present = set(handle.keys())
            require(present == by_shard[filename],
                    "Safetensors header/index key mismatch")
            for name in present:
                entry = handle.get_slice(name)
                shape = entry.get_shape()
                require(all(isinstance(dim, int) and dim > 0 for dim in shape),
                        "Invalid indexed tensor shape")
                require(bool(entry.get_dtype()), "Missing tensor dtype")
                verified += 1
    return verified


def verify_shard_hashes(
    root: Path, shards: list[str], manifest_path: Path
) -> int:
    """Stream disk-only bytes and match pinned independent SHA-256 values."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    digests = raw.get("shards") if isinstance(raw, dict) and "shards" in raw else raw
    require(isinstance(digests, dict) and set(digests) == set(shards),
            "SHA-256 manifest must cover exactly the indexed shard set")
    require(all(isinstance(v, str) and SHA256_RE.fullmatch(v.lower()) for v in digests.values()),
            "Invalid SHA-256 manifest")
    for filename in shards:
        actual = hashlib.sha256()
        with (root / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                actual.update(chunk)
        require(hmac.compare_digest(actual.hexdigest(), digests[filename].lower()),
                "Pinned shard SHA-256 mismatch")
    return len(shards)


def verify_tokenizer_mapping(root: Path, expected: str | None) -> bool:
    require(isinstance(expected, str) and SHA256_RE.fullmatch(expected.lower()),
            "Independently pinned official GLM token-to-ID SHA-256 is required")
    filename = root / "tokenizer.json"
    require(filename.is_file(), "GLM tokenizer.json is not locally staged")
    # The tokenizers Rust runtime exposes full token IDs including added tokens.
    # The reference fingerprint uses the same normalized format as
    # iq_transfer/glm_first_tokenizer.py, without importing torch/iq_transfer.
    from tokenizers import Tokenizer

    vocab = Tokenizer.from_file(str(filename)).get_vocab(with_added_tokens=True)
    require(bool(vocab) and len(vocab) == 154880,
            "GLM token map must contain exactly 154880 entries")
    ids = list(vocab.values())
    require(len(set(ids)) == len(ids) and set(ids) == set(range(154880)),
            "GLM token IDs must be a unique contiguous 0..154879 mapping")
    serialized = json.dumps(
        vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    actual = hashlib.sha256(serialized).hexdigest()
    require(hmac.compare_digest(actual, expected.lower()),
            "Full GLM tokenizer ID mapping differs from pinned source revision")
    return True


def execute(args: argparse.Namespace, telemetry: IQTelemetry) -> dict:
    root = args.checkpoint.resolve(strict=True)
    details: dict = {
        "status": "blocked",
        "gpu_launched": False,
        "inference_called": False,
        "checkpoint_modified": False,
        "quality_verified": False,
        "traces_exported": False,
        "findings": [],
    }
    with telemetry.stage("preflight") as span:
        with telemetry.stage("metadata"):
            result = audit_foundation(root, intent="glm53")
            if result["status"] == "blocked":
                details["findings"].extend(
                    finding.get("code", "unknown") for finding in result["findings"]
                    if finding.get("level") == "blocker"
                )
            weights, shards = index_inventory(root, args.expected_shards)
            telemetry.annotate(span, indexed_tensors=len(weights), shards=len(shards))

        try:
            with telemetry.stage("tokenizer") as token_span:
                verify_tokenizer_mapping(
                    args.tokenizer_dir.resolve() if args.tokenizer_dir else root,
                    args.expected_glm_tokenizer_sha256,
                )
                telemetry.annotate(token_span, tokenizer_verified=True)
        except (OSError, ValueError, ImportError) as exc:
            # The OTel span is ERROR, with exception type only.
            details["findings"].append("tokenizer_" + type(exc).__name__)

        try:
            with telemetry.stage("tensor_headers") as tensor_span:
                verified = verify_headers(root, weights, shards)
                telemetry.annotate(tensor_span, verified_tensors=verified)
                details["tensor_headers_verified"] = verified
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            details["findings"].append("tensor_headers_" + type(exc).__name__)
            details["tensor_headers_verified"] = 0

        if args.sha256_manifest:
            try:
                with telemetry.stage("tensor_hashes") as hash_span:
                    verified_hashes = verify_shard_hashes(
                        root, shards, args.sha256_manifest
                    )
                    details["hashes_verified"] = verified_hashes
                    telemetry.annotate(hash_span, hashes_verified=True)
            except (OSError, ValueError) as exc:
                details["findings"].append("tensor_hashes_" + type(exc).__name__)
                details["hashes_verified"] = 0
        else:
            details["hashes_verified"] = 0
            details["findings"].append("independent_shard_hashes_not_checked")

        details["status"] = (
            "metadata_consistent_quality_unverified"
            if not details["findings"] else "blocked"
        )
        telemetry.annotate(span, success=details["status"] != "blocked")
    return details


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--tokenizer-dir", type=Path)
    p.add_argument("--expected-glm-tokenizer-sha256",
                   help="Fingerprint from independently pinned official GLM revision")
    p.add_argument("--expected-shards", type=int, default=79)
    p.add_argument("--sha256-manifest", type=Path,
                   help="Independent {shards:{filename: sha256}} file; streams all bytes")
    p.add_argument("--telemetry", choices=("console", "otlp"), default="console")
    args = p.parse_args(argv)
    try:
        telemetry = IQTelemetry(mode=args.telemetry)
    except (RuntimeError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "blocked", "reason": type(exc).__name__,
                          "gpu_launched": False}), file=sys.stderr)
        return 2

    try:
        try:
            details = execute(args, telemetry)
        except (OSError, ValueError, ImportError, RuntimeError, KeyError) as exc:
            details = {
                "status": "blocked",
                "findings": ["preflight_" + type(exc).__name__],
                "gpu_launched": False,
                "inference_called": False,
                "quality_verified": False,
            }
        # Export result != collector receipt; do not claim delivery here.
        print(json.dumps(details, sort_keys=True))
        return 0 if details["status"] == "metadata_consistent_quality_unverified" else 2
    finally:
        telemetry.flush()
        telemetry.close()


if __name__ == "__main__":
    raise SystemExit(main())
