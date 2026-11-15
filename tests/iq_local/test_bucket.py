import os

import pytest

from iq_local.bucket import (
    BUCKET_ID,
    RUN_PREFIX,
    BucketAuthError,
    BucketClient,
    BucketObjectMissing,
)


def test_constants_match_checkpoint_run():
    assert BUCKET_ID == "kofecheckmate/iq-glm53-all256"
    assert RUN_PREFIX == "IQ/runs/glm53-full78-local-20261025a"


def test_missing_token_raises_bucket_auth_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    with pytest.raises(BucketAuthError):
        client.metadata("COMPILE_COMPLETE.json")


@pytest.mark.net
def test_metadata_and_download_real_file(tmp_path):
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    meta = client.metadata("COMPILE_COMPLETE.json")
    assert meta.size > 0
    dest = tmp_path / "iq_compile_complete_probe.json"
    client.download([("COMPILE_COMPLETE.json", dest)])
    assert dest.read_bytes()[:1] == b"{"


@pytest.mark.net
def test_exists_false_for_absent_path():
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    assert not client.exists("definitely/not/here.safetensors")
