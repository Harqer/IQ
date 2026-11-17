"""No-cost, no-weight 235-layer full-IQ Baseten preflight."""
from __future__ import annotations
import json

from scripts.stage_iq235_baseten import stage_sources

from iq_cloud.full_iq235 import (
    EXPECTED_TYPES, EXPECTED_VOCAB, GLM_REVISION,
    GPU_COUNT, physical_layer_placement, read_architecture_contract,
)


def main():
    config = read_architecture_contract()
    ids = physical_layer_placement(config)
    stage = stage_sources()
    counts = {key: config["schedule"]["layers"].count(key)
              for key in sorted(EXPECTED_TYPES)}
    print(json.dumps({
        "stage": "architecture_only",
        "full_model": True,
        "physical_layers": len(ids),
        "component_counts": counts,
        "experts_per_routed_layer": config["stable_moe"]["num_experts"],
        "mamba_mimo_rank": config["mamba3"]["mimo_rank"],
        "hidden_size": config["model"]["hidden_size"],
        "glm_tokenizer_vocab": EXPECTED_VOCAB,
        "glm_tokenizer_revision": GLM_REVISION,
        "planned_baseten_gpu_count": GPU_COUNT,
        "layer_assignments": {str(d): ids.count(d) for d in range(GPU_COUNT)},
        "gpu_started": False,
        "donor_weights_loaded": False,
        "package_stage": stage,
    }, indent=2))


if __name__ == "__main__":
    main()
