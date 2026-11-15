import os

import pytest

import httpx
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from iq_local.bucket import (
    BUCKET_ID,
    RUN_PREFIX,
    BucketAuthError,
    BucketClient,
    BucketObjectMissing,
)


def _hub_http_error(cls, status_code: int) -> HfHubHTTPError:
    response = httpx.Response(
        status_code,
        request=httpx.Request("GET", "https://huggingface.co/probe"),
    )
    return cls("hub error", response=response)


def test_constants_match_checkpoint_run():
    assert BUCKET_ID == "kofecheckmate/iq-glm53-all256"
    assert RUN_PREFIX == "IQ/runs/glm53-full78-local-20261025a"


def test_missing_token_raises_bucket_auth_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    with pytest.raises(BucketAuthError):
        client.metadata("COMPILE_COMPLETE.json")


def test_metadata_repository_not_found_maps_to_missing(monkeypatch):
    import huggingface_hub

    def raise_repo_not_found(*args, **kwargs):
        raise _hub_http_error(RepositoryNotFoundError, 404)

    monkeypatch.setenv("HF_TOKEN", "dummy-test-token")
    monkeypatch.setattr(
        huggingface_hub, "get_bucket_file_metadata", raise_repo_not_found
    )
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    with pytest.raises(BucketObjectMissing):
        client.metadata("COMPILE_COMPLETE.json")


def test_metadata_hf_http_error_classified_by_status_code(monkeypatch):
    import huggingface_hub

    def raise_unauthorized(*args, **kwargs):
        raise _hub_http_error(HfHubHTTPError, 401)

    monkeypatch.setenv("HF_TOKEN", "dummy-test-token")
    monkeypatch.setattr(
        huggingface_hub, "get_bucket_file_metadata", raise_unauthorized
    )
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    with pytest.raises(BucketAuthError):
        client.metadata("COMPILE_COMPLETE.json")


def test_metadata_non_http_error_uses_substring_fallback(monkeypatch):
    import huggingface_hub

    def raise_transport_error(*args, **kwargs):
        raise RuntimeError("connection reset by peer id=401")

    monkeypatch.setenv("HF_TOKEN", "dummy-test-token")
    monkeypatch.setattr(
        huggingface_hub, "get_bucket_file_metadata", raise_transport_error
    )
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    with pytest.raises(BucketAuthError):
        client.metadata("COMPILE_COMPLETE.json")


@pytest.mark.net
@pytest.mark.filterwarnings("ignore:hf_xet.download_files:DeprecationWarning")
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
