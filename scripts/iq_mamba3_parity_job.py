#!/usr/bin/env python3
"""Mamba-3 CPU-vs-CUDA parity fixture job (billed verification gate).

Default mode is the local preflight: no GPU and no cost. It wires the CPU
port (iq_model/state/mamba3_cpu.py, Task 7) against itself on a tiny
synthetic schedule, asserts exact-zero diffs, and writes a summary skeleton
under --work-dir.

--execute is the user-triggered H200 job mode (cost authorization follows the
scripts/iq_hf_inference.py conventions: explicit flag, explicit guards). It
loads real checkpoint weights for the selected layers via the production CUDA
runtime (Mamba3MIMOState), runs production prefill/decode, then runs the
Mamba3CPU port (looped steps) on the same seeded inputs, and records
per-tensor max_abs_diff fixtures: prefill_l{L}.json / decode_l{L}.json for
layers {0, first mamba layer after an attention block}.

Tolerance: bf16 inputs through fp32 math are expected to stay under 1e-3
max_abs_diff for outputs and under 1e-3 for state tensors after 64 prefill
steps. Exact-zero is only expected in preflight (CPU port vs itself); the
printed table reports PASS/FAIL per tensor against this tolerance.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_CHECKPOINT_DIR = REPO / "checkpoints" / "iq"
OUTPUT_TOLERANCE = 1e-3
STATE_TOLERANCE = 1e-3
ATTENTION_TYPES = {"csa", "hca", "dense_attention"}
PREFILL_BATCH, PREFILL_SEQ = 4, 64
DECODE_BATCH, DECODE_PRIME_SEQ = 1, 8
STATE_NAMES = ("angle", "ssm", "k", "v")


def _tiny_cpu_config():
    from iq_model.state.mamba3_cpu import CPUConfig

    return CPUConfig(
        d_model=32,
        headdim=16,
        d_state=8,
        mimo_rank=2,
        num_layers=2,
        expand=2.0,
        chunk_size=16,
        rope_fraction=0.5,
        outproj_norm=False,
    )


def _new_cpu_pair(config, layer_idx: int):
    from iq_model.state.mamba3_cpu import Mamba3CPU

    reference = Mamba3CPU(
        config,
        layer_idx=layer_idx,
        dtype=torch.bfloat16,
        device="cpu",
    )
    candidate = Mamba3CPU(
        config,
        layer_idx=layer_idx,
        dtype=torch.bfloat16,
        device="cpu",
    )
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return reference, candidate


def _encode_tensor(tensor: torch.Tensor) -> dict:
    cast = tensor.detach().to(torch.float32).contiguous().cpu()
    return {
        "dtype": "float32",
        "shape": list(cast.shape),
        "data": base64.b64encode(cast.numpy().tobytes()).decode("ascii"),
    }


def _decode_tensor(payload: dict) -> torch.Tensor:
    raw = base64.b64decode(payload["data"])
    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    return flat.reshape(payload["shape"])


def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _comparison(tensor: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    return {
        "tensor": tensor,
        "shape": list(a.shape),
        "max_abs_diff": _diff(a, b),
    }


def _state_payload(state) -> dict:
    return {name: _encode_tensor(getattr(state, name)) for name in STATE_NAMES}


def _state_comparisons(reference, candidate) -> list[dict]:
    return [
        _comparison(f"state.{name}", getattr(reference, name), getattr(candidate, name))
        for name in STATE_NAMES
    ]


def _production_state_comparisons(state_tuple, cpu_state) -> list[dict]:
    return [
        _comparison(f"state.{name}", tensor.detach().cpu(), getattr(cpu_state, name))
        for name, tensor in zip(STATE_NAMES, state_tuple, strict=True)
    ]


def _verdict(comparisons: list[dict], tolerance: float) -> str:
    return "pass" if all(c["max_abs_diff"] <= tolerance for c in comparisons) else "fail"


def _case_payload(case: str, layer_idx: int, input_shape, input_tensor, output_tensor,
                  state, comparisons: list[dict]) -> dict:
    return {
        "layer": layer_idx,
        "case": case,
        "input_shape": list(input_shape),
        "input": _encode_tensor(input_tensor),
        "output": _encode_tensor(output_tensor),
        "state": state,
        "comparisons": comparisons,
    }


def _write_case(work_dir: Path, payload: dict) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{payload['case']}_l{payload['layer']}.json"
    path.write_text(json.dumps(payload))
    return path


def _print_table(title: str, cases: list[dict], tolerance: float) -> None:
    print(f"\n{title} (tolerance {tolerance:.1e})")
    print(f"{'tensor':32s} {'shape':24s} {'max_abs_diff':>14s} {'verdict':>8s}")
    for case in cases:
        for record in case["comparisons"]:
            print(
                f"{record['tensor']:32s} {str(record['shape']):24s} "
                f"{record['max_abs_diff']:14.6e} {case['verdict']:>8s}"
            )


def preflight(work_dir: Path, seed_base: int) -> dict:
    """Local preflight: CPU port vs itself, exact-zero diffs, no GPU, no cost."""
    config = _tiny_cpu_config()
    reference, candidate = _new_cpu_pair(config, layer_idx=0)

    torch.manual_seed(seed_base)
    prefill_x = torch.randn(
        PREFILL_BATCH, PREFILL_SEQ, config.d_model, dtype=torch.float32
    ).to(torch.bfloat16)
    y_ref, s_ref = reference.prefill(prefill_x)
    y_cand, s_cand = candidate.prefill(prefill_x)
    prefill_comparisons = [
        _comparison("output", y_ref, y_cand),
        *_state_comparisons(s_ref, s_cand),
    ]
    prefill_payload = _case_payload(
        "prefill",
        0,
        (PREFILL_BATCH, PREFILL_SEQ),
        prefill_x,
        y_ref,
        _state_payload(s_ref),
        prefill_comparisons,
    )

    torch.manual_seed(seed_base + 1)
    reference, candidate = _new_cpu_pair(config, layer_idx=0)
    prime_x = torch.randn(
        DECODE_BATCH, DECODE_PRIME_SEQ, config.d_model, dtype=torch.float32
    ).to(torch.bfloat16)
    _, s_ref = reference.prefill(prime_x)
    _, s_cand = candidate.prefill(prime_x)
    decode_x = torch.randn(
        DECODE_BATCH, 1, config.d_model, dtype=torch.float32
    ).to(torch.bfloat16)
    y_ref, s_ref = reference.step(decode_x.reshape(DECODE_BATCH, config.d_model), s_ref)
    y_cand, s_cand = candidate.step(decode_x.reshape(DECODE_BATCH, config.d_model), s_cand)
    decode_comparisons = [
        _comparison("output", y_ref, y_cand),
        *_state_comparisons(s_ref, s_cand),
    ]
    decode_payload = _case_payload(
        "decode",
        0,
        (DECODE_BATCH, 1),
        decode_x,
        y_ref.reshape(DECODE_BATCH, 1, config.d_model),
        _state_payload(s_ref),
        decode_comparisons,
    )

    cases = [
        {
            "name": "prefill",
            "layer": 0,
            "input_shape": [PREFILL_BATCH, PREFILL_SEQ],
            "comparisons": prefill_comparisons,
            "verdict": _verdict(prefill_comparisons, OUTPUT_TOLERANCE),
        },
        {
            "name": "decode",
            "layer": 0,
            "input_shape": [DECODE_BATCH, 1],
            "comparisons": decode_comparisons,
            "verdict": _verdict(decode_comparisons, OUTPUT_TOLERANCE),
        },
    ]
    _write_case(work_dir, prefill_payload)
    _write_case(work_dir, decode_payload)
    summary = {
        "mode": "preflight",
        "tolerance": OUTPUT_TOLERANCE,
        "cases": cases,
    }
    _print_table("Mamba-3 preflight summary (CPU port vs itself)", cases, OUTPUT_TOLERANCE)
    return summary


def select_parity_layers(schedule) -> list[int]:
    """Pick {0, first mamba layer after an attention block} from the schedule."""
    layers = [str(getattr(entry, "value", entry)) for entry in schedule]
    if not layers or layers[0] != "mamba3":
        raise SystemExit(f"schedule must start with mamba3, got {layers[:1]}")
    selected = [0]
    for index in range(1, len(layers)):
        if layers[index] == "mamba3" and any(
            layers[j] in ATTENTION_TYPES for j in range(1, index)
        ):
            selected.append(index)
            break
    if len(selected) != 2:
        raise SystemExit("schedule contains no mamba layer after an attention block")
    return selected


def load_layer_weights(checkpoint_dir: Path, layer_idx: int) -> dict:
    """Load `layers.{L}.mamba.core.*` weights from the sharded checkpoint."""
    from safetensors.torch import load_file

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing checkpoint index: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = f"layers.{layer_idx}.mamba.core."
    wanted = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if not wanted:
        raise SystemExit(f"no mamba core weights for layer {layer_idx} in index")
    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        shard_path = checkpoint_dir / shard
        if not shard_path.is_file():
            raise SystemExit(f"missing checkpoint shard: {shard_path}")
        for key, tensor in load_file(str(shard_path), device="cpu").items():
            if key in wanted:
                weights["core." + key[len(prefix):]] = tensor
    loaded = {
        f"layers.{layer_idx}.mamba.core.{key[len('core.'):]}": None
        for key in weights
    }
    missing = sorted(set(wanted) - set(loaded))
    if missing:
        raise SystemExit(f"shards did not provide {missing[:5]}")
    return weights


def _assert_h200() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("H200 parity execution requires CUDA")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if "H200" not in name:
        raise SystemExit(f"H200 parity must run on H200; found {name!r}")
    from iq_model import require_mamba3_mimo_runtime

    require_mamba3_mimo_runtime(device)
    return device


def _real_mamba_configs(checkpoint_dir: Path):
    from iq_model.hybrid import IQHybridConfig
    from iq_model.state.mamba3_cpu import CPUConfig

    config = IQHybridConfig.from_json(str(checkpoint_dir / "iq_config.json"))
    mamba = config.mamba3
    cpu_config = CPUConfig(
        d_model=mamba.d_model,
        headdim=mamba.headdim,
        d_state=mamba.d_state,
        mimo_rank=mamba.mimo_rank,
        num_layers=mamba.num_layers,
        expand=mamba.expand,
        chunk_size=mamba.chunk_size,
        rope_fraction=mamba.rope_fraction,
        outproj_norm=mamba.outproj_norm,
    )
    return config.mamba3, cpu_config


def _production_step_loop(production, tokens, state) -> tuple[torch.Tensor, tuple]:
    outputs = []
    for t in range(tokens.shape[1]):
        out, angle, ssm, k, v = production.core.step(tokens[:, t], *state)
        state = (angle, ssm, k, v)
        outputs.append(out)
    return torch.stack(outputs, dim=1), state


def _production_case(production, cpu_port, device, layer_idx: int, seed: int) -> tuple[dict, dict]:
    d_model = production.config.d_model

    torch.manual_seed(seed)
    x = torch.randn(
        PREFILL_BATCH, PREFILL_SEQ, d_model,
        device=device, dtype=torch.float32,
    ).to(torch.bfloat16)
    with torch.no_grad():
        parallel = production(x)
        state = production.allocate_inference_cache(
            PREFILL_BATCH, 1, device=device, dtype=torch.bfloat16
        )
        step_output, state = _production_step_loop(production, x, state)
    cpu_y, cpu_state = cpu_port.prefill(x.detach().cpu().clone())
    prefill_comparisons = [
        _comparison("output.parallel_scan", parallel.detach().cpu(), cpu_y),
        _comparison("output.step_loop", step_output.detach().cpu(), cpu_y),
        *_production_state_comparisons(state, cpu_state),
    ]
    prefill_payload = _case_payload(
        "prefill",
        layer_idx,
        (PREFILL_BATCH, PREFILL_SEQ),
        x.detach().cpu(),
        cpu_y,
        _state_payload(cpu_state),
        prefill_comparisons,
    )

    torch.manual_seed(seed + 1000)
    prime = torch.randn(
        DECODE_BATCH, DECODE_PRIME_SEQ, d_model,
        device=device, dtype=torch.float32,
    ).to(torch.bfloat16)
    decode_x = torch.randn(
        DECODE_BATCH, 1, d_model,
        device=device, dtype=torch.float32,
    ).to(torch.bfloat16)
    with torch.no_grad():
        state = production.allocate_inference_cache(
            DECODE_BATCH, 1, device=device, dtype=torch.bfloat16
        )
        _, state = _production_step_loop(production, prime, state)
        decode_out, angle, ssm, k, v = production.core.step(decode_x[:, 0], *state)
        decode_state = (angle, ssm, k, v)
    cpu_prime_y, cpu_s = cpu_port.prefill(prime.detach().cpu().clone())
    cpu_decode_out, cpu_s = cpu_port.step(
        decode_x.detach().cpu().reshape(DECODE_BATCH, d_model),
        cpu_s,
    )
    decode_comparisons = [
        _comparison("output", decode_out.detach().cpu(), cpu_decode_out),
        *_production_state_comparisons(decode_state, cpu_s),
    ]
    decode_payload = _case_payload(
        "decode",
        layer_idx,
        (DECODE_BATCH, 1),
        decode_x.detach().cpu(),
        cpu_decode_out.reshape(DECODE_BATCH, 1, d_model),
        _state_payload(cpu_s),
        decode_comparisons,
    )
    return prefill_payload, decode_payload


def execute(work_dir: Path, seed_base: int, checkpoint_dir: Path | None = None) -> dict:
    """Billed H200 job: production CUDA runtime vs Mamba3CPU on real weights."""
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    device = _assert_h200()
    from iq_model import Mamba3MIMOState
    from iq_model.hybrid import IQHybridConfig
    from iq_model.state.mamba3_cpu import Mamba3CPU

    real_config, cpu_config = _real_mamba_configs(checkpoint)
    schedule = IQHybridConfig.from_json(str(checkpoint / "iq_config.json")).schedule.layers
    selected = select_parity_layers(schedule)

    payloads = []
    for layer_idx in selected:
        seed = seed_base + layer_idx
        production = Mamba3MIMOState(
            real_config,
            layer_idx=layer_idx,
            dtype=torch.bfloat16,
            device=device,
        )
        production.load_state_dict(load_layer_weights(checkpoint, layer_idx), strict=True)
        cpu_port = Mamba3CPU(
            cpu_config,
            layer_idx=layer_idx,
            dtype=torch.bfloat16,
            device="cpu",
        )
        cpu_port.load_state_dict(load_layer_weights(checkpoint, layer_idx), strict=True)
        prefill_payload, decode_payload = _production_case(
            production, cpu_port, device, layer_idx, seed
        )
        payloads.append(prefill_payload)
        payloads.append(decode_payload)

    for payload in payloads:
        _write_case(work_dir, payload)
    cases = [
        {
            "name": payload["case"],
            "layer": payload["layer"],
            "input_shape": payload["input_shape"],
            "comparisons": payload["comparisons"],
            "verdict": _verdict(payload["comparisons"], OUTPUT_TOLERANCE),
        }
        for payload in payloads
    ]
    summary = {
        "mode": "execute",
        "tolerance": OUTPUT_TOLERANCE,
        "cases": cases,
    }
    _print_table("Mamba-3 parity summary (production CUDA vs CPU port)", cases, OUTPUT_TOLERANCE)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true",
                        help="run the billed H200 job (cost authorization)")
    parser.add_argument("--work-dir", type=Path, default=Path("iq-transfer/parity"))
    parser.add_argument("--seed-base", type=int, default=20261025)
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="checkpoint directory for --execute (default checkpoints/iq)")
    args = parser.parse_args()
    if args.execute:
        summary = execute(
            work_dir=args.work_dir,
            seed_base=args.seed_base,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        summary = preflight(work_dir=args.work_dir, seed_base=args.seed_base)
    return 0 if all(case["verdict"] == "pass" for case in summary["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
