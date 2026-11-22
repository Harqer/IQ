import json
from pathlib import Path

import pytest

from iq_local.bucket import BUCKET_ID, RUN_PREFIX, BucketClient
from iq_local.index import RunIndex

REPO = Path(__file__).resolve().parents[2]
LOCAL_IQ = REPO / "checkpoints" / "iq"


def test_expert_stage_index_is_complete_from_local_fixture():
    run_index = RunIndex.__new__(RunIndex)
    run_index.weight_map = json.loads(
        (LOCAL_IQ / "model.safetensors.index.json").read_text()
    )["weight_map"]
    run_index.files = sorted(set(run_index.weight_map.values()))
    run_index._build_experts_by_stage()
    assert len(run_index.experts_by_stage) == 78
    for stage, experts in run_index.experts_by_stage.items():
        assert len(experts) == 256
        assert all(
            name.startswith(f"model-moe-{stage:03d}-expert-") for name in experts
        )
    assert isinstance(run_index._keys_by_file, dict)
    target = run_index.experts_by_stage[0][0]
    keys = run_index.keys_for_file(target)
    assert keys
    expected = sorted(
        key for key, file in run_index.weight_map.items() if file == target
    )
    assert len(keys) == len(expected)
    assert keys == expected
    layer = int(expected[0].split(".")[1])
    prefix = f"layers.{layer}.moe.routed_experts."
    assert all(key.startswith(prefix) for key in keys)


@pytest.mark.net
def test_run_index_downloads_and_matches_local(tmp_path):
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    run_index = RunIndex(client, tmp_path)
    assert len(run_index.weight_map) == 63164
    assert len(run_index.files) == 20283
    RunIndex.verify_against_local_checkpoint(run_index, LOCAL_IQ)
    assert run_index.stage_of_layer(2) == 0
