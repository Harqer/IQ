from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .apply import ParameterUpdate, apply_parameter_updates
from .donor import DonorInspector, OperatorRef, TensorSource
from .manifest import DonorManifest
from .plan import TransportPlan
from .provenance import ParameterProvenance, ProvenanceLedger
from .slots import TargetAssignment, TargetSlot, TransferMethod
from .transport import CoordinateMap, transport_linear


class ExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DonorRuntime:
    donor_id: str
    manifest: DonorManifest
    inspector: DonorInspector
    source: TensorSource

    def __post_init__(self) -> None:
        if not self.donor_id.strip():
            raise ExecutionError("donor_id must be non-empty")
        if self.manifest.donor_id != self.donor_id:
            raise ExecutionError(
                f"donor runtime id {self.donor_id!r} does not match manifest id {self.manifest.donor_id!r}"
            )


@dataclass(frozen=True)
class TransportExecutionReport:
    updates: tuple[ParameterUpdate, ...]
    applied_parameters: tuple[str, ...]
    deferred_functional: tuple[str, ...]
    skipped_recipe: tuple[str, ...]
    recipient_native: tuple[str, ...]
    provenance: ProvenanceLedger


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ExecutionError("source/transport tensor contains non-finite values")
    return array


def _target_parameter_path(assignment: TargetAssignment) -> str:
    return assignment.target_module_path if assignment.target_module_path.endswith(".weight") else assignment.target_module_path + ".weight"


def _operator_index(runtime: DonorRuntime) -> dict[tuple[int, str], OperatorRef]:
    report = runtime.inspector.validate_checkpoint(runtime.source)
    report.require_ok()
    result: dict[tuple[int, str], OperatorRef] = {}
    for ref in runtime.inspector.operators(runtime.source):
        key = (ref.layer, ref.role)
        if key in result:
            raise ExecutionError(f"duplicate donor operator: layer={ref.layer} role={ref.role}")
        result[key] = ref
    return result


def _source_value(
    assignment: TargetAssignment,
    runtime: DonorRuntime,
    operators: Mapping[tuple[int, str], OperatorRef],
) -> tuple[Any, str, str | None]:
    if assignment.source_operator is None:
        raise ExecutionError("donor-backed assignment is missing source_operator")
    if assignment.source_layer is None:
        if assignment.source_operator not in runtime.source.keys():
            raise ExecutionError(f"missing raw donor tensor: {assignment.source_operator}")
        return runtime.source.get(assignment.source_operator), assignment.source_operator, None

    key = (assignment.source_layer, assignment.source_operator)
    try:
        ref = operators[key]
    except KeyError as exc:
        raise ExecutionError(
            f"missing donor operator: layer={assignment.source_layer} role={assignment.source_operator}"
        ) from exc
    slice_desc = None
    if ref.row_start is not None:
        slice_desc = f"rows[{ref.row_start}:{ref.row_stop}]"
    return ref.materialize(runtime.source), ref.source_key, slice_desc


def _require_map(maps: Mapping[str, CoordinateMap], map_id: str | None, role: str) -> CoordinateMap:
    if not map_id:
        raise ExecutionError(f"{role} coordinate map id is required")
    try:
        return maps[map_id]
    except KeyError as exc:
        raise ExecutionError(f"missing coordinate map {map_id!r} for {role}") from exc


def _transport_embedding(weight: Any, output_map: CoordinateMap) -> np.ndarray:
    source = _numpy(weight)
    if source.ndim != 2:
        raise ExecutionError("embedding source tensor must be rank-2")
    if source.shape[1] != output_map.matrix.shape[0]:
        raise ExecutionError(
            f"embedding/map mismatch: source hidden={source.shape[1]}, map source={output_map.matrix.shape[0]}"
        )
    return source @ output_map.matrix


def _transport_lm_head(weight: Any, input_map: CoordinateMap) -> np.ndarray:
    source = _numpy(weight)
    if source.ndim != 2:
        raise ExecutionError("LM-head source tensor must be rank-2")
    if source.shape[1] != input_map.matrix.shape[0]:
        raise ExecutionError(
            f"LM-head/map mismatch: source hidden={source.shape[1]}, map source={input_map.matrix.shape[0]}"
        )
    return source @ np.linalg.pinv(input_map.matrix).T


def execute_transport_plan(
    model: torch.nn.Module,
    plan: TransportPlan,
    donors: tuple[DonorRuntime, ...] | list[DonorRuntime],
    coordinate_maps: Mapping[str, CoordinateMap],
    *,
    training_phase: str = "T1",
    apply: bool = True,
) -> TransportExecutionReport:
    if not training_phase.strip():
        raise ExecutionError("training_phase must be non-empty")

    config = getattr(model, "config", None)
    config_hash = getattr(config, "fingerprint", None)
    if not isinstance(config_hash, str) or config_hash != plan.iq_config_hash:
        raise ExecutionError("recipient model config fingerprint does not match the transport plan")

    runtime_by_id: dict[str, DonorRuntime] = {}
    for runtime in donors:
        if runtime.donor_id in runtime_by_id:
            raise ExecutionError(f"duplicate donor runtime: {runtime.donor_id}")
        runtime_by_id[runtime.donor_id] = runtime

    expected_donors = plan.donor_map()
    if set(runtime_by_id) != set(expected_donors):
        raise ExecutionError(
            f"donor runtime set mismatch: expected={sorted(expected_donors)}, got={sorted(runtime_by_id)}"
        )
    for donor_id, expected_fingerprint in expected_donors.items():
        actual = runtime_by_id[donor_id].manifest.fingerprint
        if actual != expected_fingerprint:
            raise ExecutionError(
                f"donor fingerprint mismatch for {donor_id}: expected {expected_fingerprint}, got {actual}"
            )

    operator_indexes = {donor_id: _operator_index(runtime) for donor_id, runtime in runtime_by_id.items()}
    updates: list[ParameterUpdate] = []
    provenance: list[ParameterProvenance] = []
    deferred_functional: list[str] = []
    skipped_recipe: list[str] = []
    recipient_native: list[str] = []

    for assignment in plan.assignments:
        target_parameter = _target_parameter_path(assignment)
        if assignment.transfer_method is TransferMethod.RECIPIENT_NATIVE:
            recipient_native.append(target_parameter)
            continue
        if assignment.transfer_method is TransferMethod.RECIPE_TRANSFER:
            skipped_recipe.append(target_parameter)
            continue
        if assignment.transfer_method is TransferMethod.FUNCTIONAL_TRANSFER:
            deferred_functional.append(target_parameter)
            continue

        if assignment.source_donor_id is None:
            raise ExecutionError(f"assignment {target_parameter} has no source donor")
        runtime = runtime_by_id[assignment.source_donor_id]
        source_value, source_tensor, source_slice = _source_value(
            assignment,
            runtime,
            operator_indexes[assignment.source_donor_id],
        )

        if assignment.transfer_method is TransferMethod.EXACT:
            value = _numpy(source_value)
            map_ids: tuple[str, ...] = ()
        elif assignment.transfer_method is TransferMethod.OPERATOR_TRANSPORT:
            if assignment.target_slot is TargetSlot.EMBEDDING:
                output_map = _require_map(coordinate_maps, assignment.output_map_id, "embedding output")
                value = _transport_embedding(source_value, output_map)
                map_ids = (assignment.output_map_id,)  # type: ignore[arg-type]
            elif assignment.target_slot is TargetSlot.LM_HEAD:
                input_map = _require_map(coordinate_maps, assignment.input_map_id, "LM-head input")
                value = _transport_lm_head(source_value, input_map)
                map_ids = (assignment.input_map_id,)  # type: ignore[arg-type]
            else:
                input_map = _require_map(coordinate_maps, assignment.input_map_id, "operator input")
                output_map = _require_map(coordinate_maps, assignment.output_map_id, "operator output")
                value = transport_linear(source_value, input_map, output_map)
                map_ids = (assignment.input_map_id, assignment.output_map_id)  # type: ignore[arg-type]
        else:
            raise ExecutionError(f"unsupported transfer method: {assignment.transfer_method.value}")

        if tuple(value.shape) != tuple(assignment.target_shape):
            raise ExecutionError(
                f"transported shape mismatch for {target_parameter}: got {tuple(value.shape)}, "
                f"plan expects {assignment.target_shape}"
            )

        updates.append(
            ParameterUpdate(
                target_parameter=target_parameter,
                value=value,
                source_description=f"{assignment.source_donor_id}:{source_tensor}",
            )
        )
        provenance.append(
            ParameterProvenance(
                target_parameter=target_parameter,
                target_slot=assignment.target_slot,
                transfer_method=assignment.transfer_method,
                training_phase_introduced=training_phase,
                source_donor_id=assignment.source_donor_id,
                source_tensor=source_tensor,
                source_slice=source_slice,
                map_ids=map_ids,
                artifact_hashes=assignment.artifact_hashes,
            )
        )

    applied_parameters: tuple[str, ...] = ()
    if apply and updates:
        applied_parameters = apply_parameter_updates(model, tuple(updates))

    return TransportExecutionReport(
        updates=tuple(updates),
        applied_parameters=applied_parameters,
        deferred_functional=tuple(deferred_functional),
        skipped_recipe=tuple(skipped_recipe),
        recipient_native=tuple(recipient_native),
        provenance=ProvenanceLedger(provenance),
    )
