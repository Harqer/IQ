from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class SlotError(RuntimeError):
    pass


class TargetSlot(str, Enum):
    RESIDUAL = "residual"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    NORM_SCALE = "norm_scale"
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_O = "attn_o"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    MAMBA3_X = "mamba3_x"
    MAMBA3_B = "mamba3_b"
    MAMBA3_C = "mamba3_c"
    MAMBA3_OUT = "mamba3_out"
    MOE_EXPERT_GATE = "moe_expert_gate"
    MOE_EXPERT_UP = "moe_expert_up"
    MOE_EXPERT_DOWN = "moe_expert_down"
    MOE_ROUTER = "moe_router"
    MTP_PROJECTION = "mtp_projection"


class TransferMethod(str, Enum):
    EXACT = "exact"
    OPERATOR_TRANSPORT = "operator_transport"
    FUNCTIONAL_TRANSFER = "functional_transfer"
    RECIPE_TRANSFER = "recipe_transfer"
    RECIPIENT_NATIVE = "recipient_native"


@dataclass(frozen=True)
class TargetAssignment:
    target_module_path: str
    target_slot: TargetSlot
    target_shape: tuple[int, ...]
    transfer_method: TransferMethod
    source_donor_id: str | None = None
    source_layer: int | None = None
    source_operator: str | None = None
    input_map_id: str | None = None
    output_map_id: str | None = None
    initialization_version: str = "1"
    artifact_hashes: tuple[str, ...] = ()
    verification_metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.target_module_path.strip():
            raise SlotError("target_module_path must be non-empty")
        if not self.target_shape or any(int(x) <= 0 for x in self.target_shape):
            raise SlotError("target_shape dimensions must be positive")
        if not self.initialization_version.strip():
            raise SlotError("initialization_version must be non-empty")

        donor_methods = {
            TransferMethod.EXACT,
            TransferMethod.OPERATOR_TRANSPORT,
            TransferMethod.FUNCTIONAL_TRANSFER,
        }
        if self.transfer_method in donor_methods:
            if not self.source_donor_id or not self.source_operator:
                raise SlotError(f"{self.transfer_method.value} requires source_donor_id and source_operator")
        if self.transfer_method is TransferMethod.OPERATOR_TRANSPORT:
            if not self.input_map_id or not self.output_map_id:
                raise SlotError("operator_transport requires input_map_id and output_map_id")
        if self.transfer_method is TransferMethod.RECIPIENT_NATIVE and any(
            value is not None
            for value in (self.source_donor_id, self.source_layer, self.source_operator, self.input_map_id, self.output_map_id)
        ):
            raise SlotError("recipient_native assignments cannot name donor tensors or coordinate maps")

        names = [name for name, _ in self.verification_metrics]
        if len(names) != len(set(names)):
            raise SlotError("verification metric names must be unique")

    @property
    def key(self) -> tuple[str, TargetSlot]:
        return self.target_module_path, self.target_slot

    def metrics(self) -> Mapping[str, float]:
        return dict(self.verification_metrics)


class TargetRegistry:
    def __init__(self, assignments: tuple[TargetAssignment, ...] | list[TargetAssignment] = ()) -> None:
        self._assignments: dict[tuple[str, TargetSlot], TargetAssignment] = {}
        for assignment in assignments:
            self.add(assignment)

    def add(self, assignment: TargetAssignment) -> None:
        if assignment.key in self._assignments:
            module, slot = assignment.key
            raise SlotError(f"duplicate target assignment: {module}:{slot.value}")
        self._assignments[assignment.key] = assignment

    def get(self, module_path: str, slot: TargetSlot) -> TargetAssignment:
        try:
            return self._assignments[(module_path, slot)]
        except KeyError as exc:
            raise SlotError(f"unassigned target slot: {module_path}:{slot.value}") from exc

    def assignments(self) -> tuple[TargetAssignment, ...]:
        return tuple(self._assignments[key] for key in sorted(self._assignments, key=lambda x: (x[0], x[1].value)))

    def require_paths(self, required: Mapping[str, tuple[TargetSlot, ...]]) -> None:
        missing: list[str] = []
        for path, slots in required.items():
            for slot in slots:
                if (path, slot) not in self._assignments:
                    missing.append(f"{path}:{slot.value}")
        if missing:
            raise SlotError(f"missing target assignments: {', '.join(missing)}")
