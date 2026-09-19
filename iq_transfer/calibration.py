from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import json


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    category: str
    content_sha256: str
    utf8_bytes: int


@dataclass(frozen=True)
class CalibrationManifest:
    name: str
    samples: tuple[CalibrationSample, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise CalibrationError(f"unsupported calibration schema: {self.schema_version}")
        if not self.name.strip():
            raise CalibrationError("calibration manifest name must be non-empty")
        if not self.samples:
            raise CalibrationError("calibration manifest must contain samples")
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise CalibrationError("calibration sample ids must be unique")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


def build_calibration_manifest(
    name: str,
    samples: Iterable[tuple[str, str, str]],
) -> CalibrationManifest:
    records: list[CalibrationSample] = []
    for sample_id, category, text in samples:
        if not sample_id.strip() or not category.strip():
            raise CalibrationError("sample id and category must be non-empty")
        if not text:
            raise CalibrationError(f"calibration sample {sample_id!r} is empty")
        encoded = text.encode("utf-8")
        records.append(
            CalibrationSample(
                sample_id=sample_id,
                category=category,
                content_sha256=sha256(encoded).hexdigest(),
                utf8_bytes=len(encoded),
            )
        )
    return CalibrationManifest(name=name, samples=tuple(records))
