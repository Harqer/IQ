from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from .slots import TargetSlot, TransferMethod


class ProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParameterProvenance:
    target_parameter: str
    target_slot: TargetSlot
    transfer_method: TransferMethod
    training_phase_introduced: str
    source_donor_id: str | None = None
    source_tensor: str | None = None
    source_slice: str | None = None
    map_ids: tuple[str, ...] = ()
    artifact_hashes: tuple[str, ...] = ()
    dora_namespace: str | None = None
    last_phase_unfrozen: str | None = None

    def __post_init__(self) -> None:
        if not self.target_parameter.strip() or not self.training_phase_introduced.strip():
            raise ProvenanceError("target_parameter and training_phase_introduced must be non-empty")
        if self.transfer_method is TransferMethod.RECIPIENT_NATIVE:
            if any(value is not None for value in (self.source_donor_id, self.source_tensor, self.source_slice)) or self.map_ids:
                raise ProvenanceError("recipient-native parameters cannot claim donor/map provenance")
        elif self.transfer_method in {
            TransferMethod.EXACT,
            TransferMethod.OPERATOR_TRANSPORT,
            TransferMethod.FUNCTIONAL_TRANSFER,
        }:
            if not self.source_donor_id or not self.source_tensor:
                raise ProvenanceError(f"{self.transfer_method.value} provenance requires donor and source tensor")


class ProvenanceLedger:
    def __init__(self, records: tuple[ParameterProvenance, ...] | list[ParameterProvenance] = ()) -> None:
        self._records: dict[str, ParameterProvenance] = {}
        for record in records:
            self.add(record)

    def add(self, record: ParameterProvenance) -> None:
        if record.target_parameter in self._records:
            raise ProvenanceError(f"duplicate provenance for target parameter: {record.target_parameter}")
        self._records[record.target_parameter] = record

    def get(self, target_parameter: str) -> ParameterProvenance:
        try:
            return self._records[target_parameter]
        except KeyError as exc:
            raise ProvenanceError(f"missing provenance for target parameter: {target_parameter}") from exc

    def records(self) -> tuple[ParameterProvenance, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def to_mapping(self) -> dict[str, Any]:
        records = []
        for record in self.records():
            item = asdict(record)
            item["target_slot"] = record.target_slot.value
            item["transfer_method"] = record.transfer_method.value
            item["map_ids"] = list(record.map_ids)
            item["artifact_hashes"] = list(record.artifact_hashes)
            records.append(item)
        return {"schema_version": 1, "records": records}

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_mapping(), sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ProvenanceLedger":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ProvenanceError(f"unsupported provenance schema: {data.get('schema_version')}")
        records = []
        for item in data["records"]:
            records.append(
                ParameterProvenance(
                    target_parameter=str(item["target_parameter"]),
                    target_slot=TargetSlot(str(item["target_slot"])),
                    transfer_method=TransferMethod(str(item["transfer_method"])),
                    training_phase_introduced=str(item["training_phase_introduced"]),
                    source_donor_id=item.get("source_donor_id"),
                    source_tensor=item.get("source_tensor"),
                    source_slice=item.get("source_slice"),
                    map_ids=tuple(str(x) for x in item.get("map_ids", ())),
                    artifact_hashes=tuple(str(x) for x in item.get("artifact_hashes", ())),
                    dora_namespace=item.get("dora_namespace"),
                    last_phase_unfrozen=item.get("last_phase_unfrozen"),
                )
            )
        return cls(records)
