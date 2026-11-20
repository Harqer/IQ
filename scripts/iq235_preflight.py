"""No-cost, no-weight 235-layer full-IQ Baseten preflight."""
from __future__ import annotations
import json

from scripts.stage_iq235_baseten import PACKAGE_NAMES, selected_sources

from iq_cloud.full_iq235 import (
    EXPECTED_TYPES, EXPECTED_VOCAB, GLM_REVISION,
    GPU_COUNT, physical_layer_placement, read_architecture_contract,
)


def main():
    config = read_architecture_contract()
    ids = physical_layer_placement(config)
    from iq_cloud.glm53_lexical import inspect_lexical_boundary
    from iq_cloud.full_iq235 import config_paths
    source_path, target_path, _ = config_paths()
    lexical = inspect_lexical_boundary(
        json.loads(source_path.read_text(encoding="utf-8")),
        json.loads(target_path.read_text(encoding="utf-8")),
    )
    # Metadata preflight must never replace an existing generated staging tree.
    checked_sources = selected_sources()
    stage = {
        "source_files_verified": len(checked_sources),
        "archive_package_roots": list(PACKAGE_NAMES),
        "staging_performed": False,
        "gpu_started": False,
    }
    counts = {key: config["schedule"]["layers"].count(key)
              for key in sorted(EXPECTED_TYPES)}
    print(json.dumps({
        "stage": "architecture_only",
        "lexical_boundary": lexical,
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
