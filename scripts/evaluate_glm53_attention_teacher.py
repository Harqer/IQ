#!/usr/bin/env python3
"""Evaluate one official pretrained GLM-5.3 MLA/DSA layer on source activations.

Inputs are *actual* layer-input activations from the same donor (HF format),
not IQ hidden states or randomly generated placeholders. This tool is a
source-teacher evaluation gate, not a model conversion.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from iq_transfer.glm53_official_attention import (
    load_official_glm53_attention_layer,
    official_attention_prefill,
)
from iq_transfer.glm53_parallel_contract import (
    GLM53Topology,
)


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True,
                        help="local GLM-5.3 BF16 HF model directory")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--source-activations", type=Path, required=True,
                        help="safetensors with source_layer_input [batch, tokens, 6144]")
    parser.add_argument("--shared-indices", type=Path,
                        help="for shared-indexer layers: genuine source full-indexer output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if not 0 <= args.layer < 78:
        parser.error("layer must be 0..77")
    if args.output.exists():
        parser.error("refusing to overwrite prior MLA/DSA source evaluation")
    config = json.loads((args.donor / "config.json").read_text(encoding="utf-8"))
    topology = GLM53Topology.from_hf_config(config)
    is_shared = topology.indexer_types[args.layer] == "shared"
    if is_shared != (args.shared_indices is not None):
        parser.error("a shared layer needs saved source top-k indices; a full layer does not")
    from safetensors.torch import load_file, save_file
    activations = load_file(str(args.source_activations), device="cpu")
    if set(activations) != {"source_layer_input", "token_ids"}:
        parser.error("captured source must contain source_layer_input and token_ids")
    input_tensor = activations["source_layer_input"]
    token_ids = activations["token_ids"]
    if token_ids.shape != input_tensor.shape[:2] or token_ids.dtype not in (
        torch.int32, torch.int64
    ):
        parser.error("token_ids must match source-layer [batch,sequence] positions")
    if (token_ids < 0).any():
        parser.error("source token_ids cannot be negative")
    if input_tensor.ndim != 3 or input_tensor.shape[-1] != 6144:
        parser.error("source activations must have shape [batch, seq, 6144]")
    if not torch.isfinite(input_tensor).all():
        parser.error("source activations contain nonfinite data")
    index_tensor = None
    if args.shared_indices is not None:
        indices = load_file(str(args.shared_indices), device="cpu")
        if "dsa_indices" not in indices:
            parser.error("shared-index file must contain dsa_indices")
        index_tensor = indices["dsa_indices"]
        if (index_tensor.ndim != 3 or
            index_tensor.shape[:2] != input_tensor.shape[:2]):
            parser.error("source DSA indices must cover the same batch and token positions")
        if index_tensor.dtype not in (torch.int32, torch.int64):
            parser.error("source DSA indices must be integers")
    # Official single-stage donor module, not entire pretrained GLM model.
    teacher = load_official_glm53_attention_layer(
        args.donor, layer=args.layer, device=args.device,
        dtype=torch.bfloat16,
    )
    teacher_input = input_tensor.to(args.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output, selected = official_attention_prefill(
            teacher, teacher_input,
            previous_indices=index_tensor.to(args.device) if index_tensor is not None else None,
        )
    if selected is None or not bool(torch.isfinite(output).all()):
        raise RuntimeError("official attention failed to return finite output and top-k indices")
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "reference.safetensors"
    save_file({
        "source_attention_output": output.cpu().contiguous(),
        "dsa_indices": selected.cpu().to(torch.int32).contiguous(),
        "token_ids": token_ids.to(torch.int64).cpu().contiguous(),
    }, str(result_path))
    metadata = {
        "format": "iq_source_native_glm53_attention_reference_v1",
        "layer": args.layer,
        "donor_operator": "transformers==5.15.0 GlmMoeDsaAttention",
        "indexer_kind": topology.indexer_types[args.layer],
        "index_source_block": topology.index_source_blocks[args.layer],
        "donor_config_sha256": digest(args.donor / "config.json"),
        "source_activation_sha256": digest(args.source_activations),
        "token_alignment_provenance": "captured_source_token_ids",
        "source_indices_sha256": (
            digest(args.shared_indices) if args.shared_indices else None
        ),
        "output_sha256": digest(result_path),
        "iq_attention_converted": False,
        "teacher_student_parity_passed": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
