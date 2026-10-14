"""Megatron Bridge donor import integration for IQ's 78-block pipeline.

The official Bridge converts each *supported donor* to a Megatron Core Hybrid
checkpoint. A donor checkpoint is NOT directly an IQ fused-block checkpoint:
Mamba-2 vs rank-4 Mamba-3, CSA/HCA, expert size, and lexical alignment still
require IQ's explicit transfer and distillation passes.

CLI interface: NVIDIA's scripts/conversion/convert.sh import, not an
invented generic converter. See docs/NVIDIA_MEGATRON_MSC.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import json
import re
import subprocess

NEMOTRON_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16"
NEMOTRON_REVISION = "d51eab0d1f979ebc26b546e634a04f450d99158e"
GLM53_ID = "zai-org/GLM-5.3"
GLM53_REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"

DONOR_PINS: dict[str, tuple[str, str]] = {
    "nemotron-super": (NEMOTRON_ID, NEMOTRON_REVISION),
    "glm53": (GLM53_ID, GLM53_REVISION),
}


class MegatronBridgeIntegrationError(ValueError):
    pass


@dataclass(frozen=True)
class MegatronImportPlan:
    donor: str
    official_repo: str
    official_revision: str
    source: str
    output_dir: str
    megatron_bridge_root: str
    device: str
    command: tuple[str, ...]
    iq_target_blocks: int
    native_donor_to_iq_equivalence: bool
    requires_iq_operator_mapping: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["command"] = list(self.command)
        return data


def make_import_plan(
    *,
    donor: Literal["nemotron-super", "glm53"],
    source: str,
    output_dir: str,
    bridge_root: str,
    device: Literal["cpu", "gpu"],
    executor: Literal["local", "slurm"] = "local",
    gpus_per_node: int = 0,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    expert_tensor_parallel: int = 1,
    nodes: int = 1,
    cpu_processes_per_node: int = 1,
    distributed_timeout_minutes: int = 60,
) -> MegatronImportPlan:
    if donor not in DONOR_PINS:
        raise MegatronBridgeIntegrationError("unknown official Bridge donor")
    if device not in ("cpu", "gpu") or executor not in ("local", "slurm"):
        raise MegatronBridgeIntegrationError("unsupported Bridge device or executor")
    if not source or not output_dir or not bridge_root:
        raise MegatronBridgeIntegrationError("source, output_dir and bridge_root are required")
    if source.startswith(("msc://", "hf://", "s3://", "gs://")):
        raise MegatronBridgeIntegrationError(
            "Bridge conversion requires a Hub ID or a mounted/local HF directory; "
            "direct object-store URIs are for Megatron Core MSC checkpointing"
        )
    official_repo, revision = DONOR_PINS[donor]
    source_is_local = source.startswith("/")
    if not source_is_local and source != official_repo:
        raise MegatronBridgeIntegrationError(
            "Hub source must be the pinned donor repo; use an absolute path for mounted weights"
        )
    if source_is_local:
        donor_root = Path(source)
        if not (donor_root.is_dir() and (donor_root / "config.json").is_file()):
            raise MegatronBridgeIntegrationError(
                "mounted donor root must contain its actual Hugging Face config.json"
            )
    output = Path(output_dir)
    bridge = Path(bridge_root)
    if not output.is_absolute() or not bridge.is_absolute():
        raise MegatronBridgeIntegrationError("Bridge root and checkpoint output must be absolute paths")
    if output.exists() and any(output.iterdir()):
        raise MegatronBridgeIntegrationError(
            "output checkpoint directory is nonempty; refuse to overwrite an existing conversion"
        )
    if output.resolve() == bridge.resolve() or (
        source_is_local and output.resolve() == Path(source).resolve()
    ):
        raise MegatronBridgeIntegrationError("source and output directories must be distinct")
    ranks = (tensor_parallel, pipeline_parallel, expert_parallel, expert_tensor_parallel)
    if any(not isinstance(i, int) or i <= 0 for i in (*ranks, nodes, cpu_processes_per_node, distributed_timeout_minutes)):
        raise MegatronBridgeIntegrationError("all rank, node and timeout counts must be positive integers")
    if device == "gpu" and gpus_per_node <= 0:
        raise MegatronBridgeIntegrationError("GPU conversion requires explicit gpus_per_node")
    if device == "cpu" and gpus_per_node != 0:
        raise MegatronBridgeIntegrationError("CPU conversion must not request GPUs")
    if device == "gpu" and executor == "local" and nodes != 1:
        raise MegatronBridgeIntegrationError("multi-node conversion requires a Slurm executor")
    if device == "cpu" and executor == "local" and nodes != 1:
        raise MegatronBridgeIntegrationError("local CPU conversion must be one node")
    command = [
        str(bridge / "scripts/conversion/convert.sh"), "import",
        "--executor", executor, "--device", device,
        "--nodes", str(nodes),
        "--hf-model", source,
        "--megatron-path", str(output),
        "--torch-dtype", "bfloat16",
        "--tp", str(tensor_parallel),
        "--pp", str(pipeline_parallel),
        "--ep", str(expert_parallel),
        "--etp", str(expert_tensor_parallel),
        "--distributed-timeout-minutes", str(distributed_timeout_minutes),
    ]
    if not source_is_local:
        command += ["--hf-revision", revision]
    if device == "gpu":
        command += ["--gpus-per-node", str(gpus_per_node)]
    elif cpu_processes_per_node > 1:
        # For large CPU imports NVIDIA documents slurm/distributed CPU mode.
        if executor != "slurm":
            raise MegatronBridgeIntegrationError(
                "multi-process CPU import must use a supported Slurm executor"
            )
        command += ["--cpu-processes-per-node", str(cpu_processes_per_node)]
    return MegatronImportPlan(
        donor=donor, official_repo=official_repo,
        official_revision=revision, source=source,
        output_dir=str(output), megatron_bridge_root=str(bridge),
        device=device, command=tuple(command),
        iq_target_blocks=78, native_donor_to_iq_equivalence=False,
        requires_iq_operator_mapping=True,
    )


def probe_provider(plan: MegatronImportPlan) -> dict[str, str]:
    """Use the real NVIDIA AutoBridge on a worker where Bridge is installed.

    Config/provider only; it does NOT instantiate large model weights.
    GLM-5.3 uses GLM5Bridge's to_megatron_provider (not get_model_config).
    """
    try:
        from megatron.bridge import AutoBridge
    except ImportError as exc:
        raise MegatronBridgeIntegrationError(
            "Megatron Bridge is not installed on this worker"
        ) from exc
    kwargs = (
        {} if plan.source.startswith("/")
        else {"revision": plan.official_revision}
    )
    bridge = AutoBridge.from_hf_pretrained(plan.source, **kwargs)
    provider = bridge.to_megatron_provider(load_weights=False)
    actual = type(getattr(bridge, "_model_bridge", bridge)).__name__
    if plan.donor == "glm53" and actual != "GLM5Bridge":
        raise MegatronBridgeIntegrationError(
            f"GLM-5.3 donor must resolve to GLM5Bridge, got {actual}"
        )
    return {
        "bridge_class": actual,
        "provider_class": type(provider).__name__,
        "donor": plan.official_repo,
        "revision": plan.official_revision,
        "weights_loaded": "false",
        "iq_converted": "false",
    }


def execute_import(plan: MegatronImportPlan) -> None:
    """Execute NVIDIA's exact conversion command; do not hide a failed import."""
    conversion_script = Path(plan.command[0])
    if not conversion_script.is_file():
        raise MegatronBridgeIntegrationError(
            f"NVIDIA Megatron Bridge conversion script not found: {conversion_script}"
        )
    subprocess.run(
        list(plan.command), check=True, cwd=plan.megatron_bridge_root
    )


def render_plan(plan: MegatronImportPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n"
