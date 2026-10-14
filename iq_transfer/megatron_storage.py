"""Portable Megatron Core Multi-Storage Client (MSC) configuration for IQ.

This module emits MSC's documented YAML configuration (JSON is valid YAML 1.2).
Credentials are ALWAYS referenced through environment variable expressions.
No credentials or donor weights are fetched by rendering storage profiles.

Documentation:
https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/msc_integration.html
https://nvidia.github.io/multi-storage-client/references/configuration.html
https://huggingface.co/docs/hub/storage-buckets-s3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Mapping, Any

MOCHA_BUCKET_URI = "hf://buckets/kofecheckmate/mocha"
MOCHA_MSC_URI = "msc://mocha/"
HF_GATEWAY = "https://s3.hf.co/kofecheckmate"
_ALLOWED_TYPES = frozenset({
    "s3", "gcs", "gcs_s3", "azure", "oci", "ais", "ais_s3", "s8k",
})
_PROFILE_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}\Z")


class MegatronStorageError(ValueError):
    pass


@dataclass(frozen=True)
class StorageProfile:
    name: str
    provider: str
    bucket: str
    endpoint_url: str | None = None
    region_name: str | None = None
    project_id: str | None = None
    namespace: str | None = None
    # Name of env variables, never the credential values themselves.
    access_key_env: str | None = None
    secret_key_env: str | None = None
    extra_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _PROFILE_RE.fullmatch(self.name) or self.name.startswith("_"):
            raise MegatronStorageError(f"invalid MSC profile name: {self.name!r}")
        if self.provider not in _ALLOWED_TYPES:
            raise MegatronStorageError(f"unsupported MSC provider: {self.provider}")
        if not self.bucket or "/" in self.bucket or self.bucket.strip() != self.bucket:
            raise MegatronStorageError("MSC base_path must be one bucket/container name")
        if self.endpoint_url is not None and not self.endpoint_url.startswith("https://"):
            raise MegatronStorageError("object-storage endpoint must use HTTPS")
        if (self.access_key_env is None) != (self.secret_key_env is None):
            raise MegatronStorageError("both S3 credential environment names are required")
        for env in (self.access_key_env, self.secret_key_env):
            if env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
                raise MegatronStorageError("credentials must use environment variable names")
        if self.access_key_env is not None and self.provider not in {"s3", "gcs_s3", "ais_s3", "s8k"}:
            raise MegatronStorageError("S3 credentials are only for S3-compatible profiles")
        if self.provider == "azure" and not self.endpoint_url:
            raise MegatronStorageError("Azure requires an endpoint_url")
        if self.provider == "oci" and not self.namespace:
            raise MegatronStorageError("OCI requires a namespace")
        if self.provider == "gcs" and not self.project_id:
            raise MegatronStorageError("GCS requires a project_id")
        if not isinstance(self.extra_options, Mapping):
            raise MegatronStorageError("extra_options must be a mapping")
        reserved = {"base_path", "endpoint_url", "region_name", "project_id", "namespace", "s3", "request_checksum_calculation", "response_checksum_validation"}
        if set(self.extra_options) & reserved:
            raise MegatronStorageError("reserved MSC provider options must not be overridden")

    def to_msc_entry(self) -> dict[str, Any]:
        options: dict[str, Any] = {"base_path": self.bucket}
        for field_name in ("endpoint_url", "region_name", "project_id", "namespace"):
            value = getattr(self, field_name)
            if value is not None:
                options[field_name] = value
        options.update(self.extra_options)
        entry: dict[str, Any] = {
            "storage_provider": {"type": self.provider, "options": options}
        }
        if self.access_key_env is not None:
            entry["credentials_provider"] = {
                "type": "S3Credentials",
                "options": {
                    "access_key": "${" + self.access_key_env + "}",
                    "secret_key": "${" + self.secret_key_env + "}",
                },
            }
        # HF's gateway requires path-style S3 and disables optional checksums.
        if self.name == "mocha":
            if self.provider != "s3" or self.bucket != "mocha" or self.endpoint_url != HF_GATEWAY:
                raise MegatronStorageError("mocha must point to the official HF gateway")
            options["region_name"] = "us-east-1"
            options["s3"] = {"addressing_style": "path"}
            options["request_checksum_calculation"] = "when_required"
            options["response_checksum_validation"] = "when_required"
        return entry


def mocha_profile() -> StorageProfile:
    """HF bucket's verified S3-compatible endpoint; requires generated HF S3 keys."""
    return StorageProfile(
        name="mocha", provider="s3", bucket="mocha",
        endpoint_url=HF_GATEWAY, region_name="us-east-1",
        access_key_env="HF_S3_ACCESS_KEY", secret_key_env="HF_S3_SECRET_KEY",
    )


def msc_uri(profile: str, key: str) -> str:
    if not _PROFILE_RE.fullmatch(profile) or profile.startswith("_"):
        raise MegatronStorageError("invalid MSC profile")
    if not key or key.startswith("/") or "\\" in key or "//" in key:
        raise MegatronStorageError("invalid MSC object path")
    if any(segment in (".", "..") for segment in key.split("/")):
        raise MegatronStorageError("object paths cannot traverse directories")
    return f"msc://{profile}/{key}"


def render_msc_config(profiles: list[StorageProfile], *, cache_dir: str, cache_size: str = "500G") -> str:
    if not profiles or len({item.name for item in profiles}) != len(profiles):
        raise MegatronStorageError("require distinct MSC profiles")
    path = Path(cache_dir)
    if not path.is_absolute() or ".." in path.parts:
        raise MegatronStorageError("MSC cache directory must be absolute")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT])", cache_size):
        raise MegatronStorageError("invalid MSC cache size")
    document = {
        "profiles": {p.name: p.to_msc_entry() for p in profiles},
        "cache": {"size": cache_size, "location": str(path)},
    }
    # JSON is YAML 1.2; MSC's YAML loader accepts it and no YAML writer
    # dependency is needed for the transfer code path.
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def load_profile_spec(path: str | Path) -> list[StorageProfile]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        specs = document["profiles"]
        if not isinstance(specs, list):
            raise TypeError("profiles must be a list")
        return [StorageProfile(**entry) for entry in specs]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MegatronStorageError(f"invalid IQ storage profile specification: {exc}") from exc
