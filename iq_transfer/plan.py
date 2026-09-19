from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json

from .slots import TargetAssignment, TargetRegistry, TargetSlot, TransferMethod


class PlanError(RuntimeError):
    pass


def _assignment_to_mapping(assignment: TargetAssignment) -> dict[str, Any]:
    return {
        "target_module_path": assignment.target_module_path,
        "target_slot": assignment.target_slot.value,
        "target_shape": list(assignment.target_shape),
        "transfer_method": assignment.transfer_method.value,
        "source_donor_id": assignment.source_donor_id,
        "source_layer": assignment.source_layer,
        "source_operator": assignment.source_operator,
        "input_map_id": assignment.input_map_id,
        "output_map_id": assignment.output_map_id,
        "initialization_version": assignment.initialization_version,
        "artifact_hashes": list(assignment.artifact_hashes),
        "verification_metrics": [[name, value] for name, value in assignment.verification_metrics],
    }


def _assignment_from_mapping(data: Mapping[str, Any]) -> TargetAssignment:
    return TargetAssignment(
        target_module_path=str(data["target_module_path"]),
        target_slot=TargetSlot(str(data["target_slot"])),
        target_shape=tuple(int(x) for x in data["target_shape"]),
        transfer_method=TransferMethod(str(data["transfer_method"])),
        source_donor_id=data.get("source_donor_id"),
        source_layer=int(data["source_layer"]) if data.get("source_layer") is not None else None,
        source_operator=data.get("source_operator"),
        input_map_id=data.get("input_map_id"),
        output_map_id=data.get("output_map_id"),
        initialization_version=str(data.get("initialization_version", "1")),
        artifact_hashes=tuple(str(x) for x in data.get("artifact_hashes", ())),
        verification_metrics=tuple((str(name), float(value)) for name, value in data.get("verification_metrics", ())),
    )


@dataclass(frozen=True)
class TransportPlan:
    plan_id: str
    donor_fingerprints: tuple[tuple[str, str], ...]
    assignments: tuple[TargetAssignment, ...]
    calibration_manifest_hash: str
    iq_config_hash: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PlanError(f"unsupported transport-plan schema: {self.schema_version}")
        required = {
            "plan_id": self.plan_id,
            "calibration_manifest_hash": self.calibration_manifest_hash,
            "iq_config_hash": self.iq_config_hash,
        }
        empty = [name for name, value in required.items() if not str(value).strip()]
        if empty:
            raise PlanError(f"transport-plan fields must be non-empty: {', '.join(empty)}")
        donor_ids = [donor_id for donor_id, _ in self.donor_fingerprints]
        if len(donor_ids) != len(set(donor_ids)):
            raise PlanError("donor ids must be unique")
        for donor_id, fingerprint in self.donor_fingerprints:
            if not donor_id.strip() or not fingerprint.strip():
                raise PlanError("donor ids and fingerprints must be non-empty")

        registry = TargetRegistry(list(self.assignments))
        known = set(donor_ids)
        for assignment in registry.assignments():
            if assignment.source_donor_id is not None and assignment.source_donor_id not in known:
                raise PlanError(
                    f"assignment {assignment.target_module_path}:{assignment.target_slot.value} "
                    f"references unknown donor {assignment.source_donor_id!r}"
                )

    def donor_map(self) -> dict[str, str]:
        return dict(self.donor_fingerprints)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "donor_fingerprints": [[donor, fingerprint] for donor, fingerprint in self.donor_fingerprints],
            "assignments": [_assignment_to_mapping(x) for x in self.assignments],
            "calibration_manifest_hash": self.calibration_manifest_hash,
            "iq_config_hash": self.iq_config_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "TransportPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema_version=int(data["schema_version"]),
            plan_id=str(data["plan_id"]),
            donor_fingerprints=tuple((str(a), str(b)) for a, b in data["donor_fingerprints"]),
            assignments=tuple(_assignment_from_mapping(x) for x in data["assignments"]),
            calibration_manifest_hash=str(data["calibration_manifest_hash"]),
            iq_config_hash=str(data["iq_config_hash"]),
        )
