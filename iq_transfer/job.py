from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import os
import shutil
import tempfile

import torch
from safetensors.torch import load_model, save_model

from iq_model import IQForCausalLM, IQModelConfig, install_dora

from .batches import load_token_batches
from .calibration import (
    CalibrationManifest,
    merge_coordinate_maps,
    solve_activation_pair,
    solve_layer_correspondence,
    solve_phi_layer_maps,
)
from .capture_runner import (
    ActivationBundle,
    build_phi_layer_calibration_from_bundles,
    capture_iq_activations,
    capture_phi_activations,
    load_local_phi_causal_lm,
    make_activation_pair,
)
from .donor import DonorConfig
from .executor import DonorRuntime, TransportExecutionReport, execute_transport_plan
from .manifest import DonorManifest, build_donor_manifest
from .phi4 import Phi4Inspector
from .phi_pipeline import build_phi_dense_plan_spec, build_phi_dense_transport_plan
from .plan import TransportPlan
from .provenance import ParameterProvenance, ProvenanceLedger
from .slots import TargetSlot, TransferMethod
from .transport import CoordinateMap, save_coordinate_map


class TransferJobError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhiTransferJobResult:
    output_dir: Path
    donor_manifest: DonorManifest
    transport_plan: TransportPlan
    layer_mapping: Mapping[int, int]
    dora_paths: tuple[str, ...]
    execution_report: TransportExecutionReport
    provenance: ProvenanceLedger


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "sentencepiece.bpe.model",
)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_tokenizer_files(checkpoint_dir: str | Path) -> str:
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise TransferJobError(f"checkpoint directory does not exist: {root}")
    files = [root / name for name in _TOKENIZER_FILES if (root / name).is_file()]
    if not files:
        raise TransferJobError(f"no recognized tokenizer files found in {root}")
    digest = sha256()
    for path in sorted(files, key=lambda p: p.name):
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _flatten_residuals(bundle: ActivationBundle, num_layers: int) -> dict[int, Any]:
    return {layer: bundle.require(f"layer.{layer}.residual_out") for layer in range(num_layers)}


def _dora_paths(plan: TransportPlan) -> tuple[str, ...]:
    dora_slots = {
        TargetSlot.ATTN_Q,
        TargetSlot.ATTN_K,
        TargetSlot.ATTN_V,
        TargetSlot.ATTN_O,
        TargetSlot.MLP_GATE,
        TargetSlot.MLP_UP,
        TargetSlot.MLP_DOWN,
    }
    return tuple(
        assignment.target_module_path
        for assignment in plan.assignments
        if assignment.transfer_method is TransferMethod.OPERATOR_TRANSPORT
        and assignment.target_slot in dora_slots
    )


def _final_parameter_provenance(
    model: torch.nn.Module,
    plan: TransportPlan,
    execution: TransportExecutionReport,
    dora_paths: Sequence[str],
    *,
    training_phase: str,
) -> ProvenanceLedger:
    original = {record.target_parameter: record for record in execution.provenance.records()}
    assignment_by_parameter = {
        (
            a.target_module_path
            if a.target_module_path.endswith(".weight")
            else a.target_module_path + ".weight"
        ): a
        for a in plan.assignments
    }
    dora = set(dora_paths)
    records: list[ParameterProvenance] = []

    for name, _parameter in model.named_parameters():
        matched_dora = next((path for path in dora if name.startswith(path + ".")), None)
        if matched_dora is not None:
            original_name = matched_dora + ".weight"
            assignment = assignment_by_parameter[original_name]
            if name == matched_dora + ".base.weight":
                source = original.get(original_name)
                if source is None:
                    raise TransferJobError(f"missing transported provenance for {original_name}")
                records.append(
                    ParameterProvenance(
                        target_parameter=name,
                        target_slot=source.target_slot,
                        transfer_method=source.transfer_method,
                        training_phase_introduced=source.training_phase_introduced,
                        source_donor_id=source.source_donor_id,
                        source_tensor=source.source_tensor,
                        source_slice=source.source_slice,
                        map_ids=source.map_ids,
                        artifact_hashes=source.artifact_hashes,
                        dora_namespace=matched_dora,
                    )
                )
            elif name in {
                matched_dora + ".lora_A",
                matched_dora + ".lora_B",
                matched_dora + ".magnitude",
            }:
                records.append(
                    ParameterProvenance(
                        target_parameter=name,
                        target_slot=assignment.target_slot,
                        transfer_method=TransferMethod.RECIPIENT_NATIVE,
                        training_phase_introduced=training_phase,
                        dora_namespace=matched_dora,
                    )
                )
            else:
                raise TransferJobError(f"unrecognized DoRA parameter: {name}")
            continue

        source = original.get(name)
        if source is not None:
            records.append(source)
            continue

        assignment = assignment_by_parameter.get(name)
        if assignment is not None and assignment.transfer_method is TransferMethod.RECIPIENT_NATIVE:
            records.append(
                ParameterProvenance(
                    target_parameter=name,
                    target_slot=assignment.target_slot,
                    transfer_method=TransferMethod.RECIPIENT_NATIVE,
                    training_phase_introduced=training_phase,
                )
            )
            continue
        raise TransferJobError(f"model parameter has no transfer provenance: {name}")

    ledger = ProvenanceLedger(records)
    current = {name for name, _ in model.named_parameters()}
    recorded = {record.target_parameter for record in ledger.records()}
    if current != recorded:
        raise TransferJobError(
            "final provenance coverage mismatch: "
            f"missing={sorted(current-recorded)}, extra={sorted(recorded-current)}"
        )
    return ledger


def _json_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _write_bundle(
    destination: Path,
    *,
    model: IQForCausalLM,
    config: IQModelConfig,
    donor_manifest: DonorManifest,
    calibration_manifest: CalibrationManifest,
    plan: TransportPlan,
    maps: Mapping[str, CoordinateMap],
    provenance: ProvenanceLedger,
    dora_paths: Sequence[str],
    dora_rank: int,
    dora_alpha: float | None,
    report: dict[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        save_model(model, str(temp / "model.safetensors"), force_contiguous=True)
        config.write_json(str(temp / "config.json"))
        donor_manifest.write_json(temp / "donor_manifest.json")
        calibration_manifest.write_json(temp / "calibration_manifest.json")
        plan.write_json(temp / "transport_plan.json")
        provenance.write_json(temp / "provenance.json")
        (temp / "adapter_config.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "type": "dora",
                    "paths": list(dora_paths),
                    "rank": int(dora_rank),
                    "alpha": dora_alpha,
                    "dropout": 0.0,
                    "freeze_base": True,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        maps_dir = temp / "maps"
        maps_dir.mkdir()
        map_index: dict[str, str] = {}
        for map_id in sorted(maps):
            filename = map_id.replace("/", "__").replace(".", "__")
            if filename in map_index.values():
                raise TransferJobError(f"coordinate-map filename collision for {map_id}")
            map_index[map_id] = filename
            save_coordinate_map(maps[map_id], maps_dir / filename)
        (maps_dir / "index.json").write_text(
            json.dumps({"schema_version": 1, "maps": map_index}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temp / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        files = []
        for path in sorted(p for p in temp.rglob("*") if p.is_file()):
            if path.name in {"bundle_manifest.json", "COMPLETE"}:
                continue
            files.append(
                {
                    "path": str(path.relative_to(temp)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        (temp / "bundle_manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": files}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (temp / "COMPLETE").write_text("ok\n", encoding="utf-8")

        if destination.exists():
            backup = destination.with_name(destination.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
            try:
                os.replace(temp, destination)
            except Exception:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temp, destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_transferred_iq_artifact(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> IQForCausalLM:
    root = Path(path)
    if not (root / "COMPLETE").is_file():
        raise TransferJobError(f"transferred IQ artifact is incomplete: {root}")
    config = IQModelConfig.from_json(str(root / "config.json"))
    model = IQForCausalLM(config)
    try:
        adapter = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransferJobError("adapter_config.json is missing or invalid") from exc
    if adapter.get("schema_version") != 1 or adapter.get("type") != "dora":
        raise TransferJobError("unsupported adapter configuration")
    install_dora(
        model,
        tuple(str(x) for x in adapter["paths"]),
        rank=int(adapter["rank"]),
        alpha=float(adapter["alpha"]) if adapter.get("alpha") is not None else None,
        dropout=float(adapter.get("dropout", 0.0)),
        freeze_base=bool(adapter.get("freeze_base", True)),
    )
    missing, unexpected = load_model(
        model,
        str(root / "model.safetensors"),
        strict=True,
        device=str(device),
    )
    if missing or unexpected:
        raise TransferJobError(
            f"transferred model state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device)


def run_phi_dense_transfer_loaded(
    *,
    phi_model: Any,
    iq_model: IQForCausalLM,
    donor_runtime: DonorRuntime,
    calibration_manifest: CalibrationManifest,
    fit_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    dora_rank: int = 64,
    dora_alpha: float | None = None,
    ridge: float = 1e-3,
    shadow_measurements: int = 256,
    shadow_seed: int = 0,
    training_phase: str = "T1",
) -> PhiTransferJobResult:
    if dora_rank <= 0:
        raise TransferJobError("dora_rank must be positive")
    if calibration_manifest.tokenizer_hash != donor_runtime.manifest.tokenizer_hash:
        raise TransferJobError("calibration tokenizer hash does not match donor manifest")
    if donor_runtime.manifest.vocab_size != iq_model.config.vocab_size:
        raise TransferJobError("Phi proof requires donor and IQ vocab sizes to match")

    source_config = donor_runtime.inspector.config
    if source_config.head_dim != iq_model.config.head_dim:
        raise TransferJobError(
            "current GQA transport requires equal head_dim: "
            f"donor={source_config.head_dim}, iq={iq_model.config.head_dim}"
        )
    if (
        source_config.num_attention_heads // source_config.num_key_value_heads
        != iq_model.config.kv_repeat
    ):
        raise TransferJobError(
            "current GQA transport requires equal query-heads-per-KV-group"
        )
    if iq_model.config.tie_word_embeddings:
        raise TransferJobError("first Phi proof requires untied IQ embeddings/LM head")

    source_fit = capture_phi_activations(phi_model, fit_batches)
    target_fit = capture_iq_activations(iq_model, fit_batches)
    source_validation = capture_phi_activations(phi_model, validation_batches)
    target_validation = capture_iq_activations(iq_model, validation_batches)

    correspondence = solve_layer_correspondence(
        _flatten_residuals(source_fit, source_config.num_hidden_layers),
        _flatten_residuals(target_fit, iq_model.config.num_hidden_layers),
        measurements=shadow_measurements,
        seed=shadow_seed,
    )

    embedding_map_id = "embedding"
    final_map_id = "final"
    embedding_pair = make_activation_pair(
        source_fit,
        target_fit,
        source_validation,
        target_validation,
        source_space="embedding",
        target_space="embedding",
    )
    final_pair = make_activation_pair(
        source_fit,
        target_fit,
        source_validation,
        target_validation,
        source_space="final",
        target_space="final",
    )
    maps: dict[str, CoordinateMap] = {
        embedding_map_id: solve_activation_pair(embedding_pair, ridge=ridge),
        final_map_id: solve_activation_pair(final_pair, ridge=ridge),
    }

    layer_ids = {}
    gqa_variance: dict[str, float] = {}
    for target_layer, source_layer in sorted(correspondence.mapping.items()):
        calibration = build_phi_layer_calibration_from_bundles(
            source_fit,
            target_fit,
            source_validation,
            target_validation,
            source_layer=source_layer,
            target_layer=target_layer,
        )
        solved = solve_phi_layer_maps(
            calibration,
            target_layer=target_layer,
            source_q_heads=source_config.num_attention_heads,
            source_kv_heads=source_config.num_key_value_heads,
            target_q_heads=iq_model.config.num_attention_heads,
            target_kv_heads=iq_model.config.num_key_value_heads,
            head_dim=source_config.head_dim,
            ridge=ridge,
        )
        maps = merge_coordinate_maps(maps, solved.maps)
        layer_ids[target_layer] = solved.map_ids
        gqa_variance[str(target_layer)] = solved.gqa_projection.explained_variance_ratio

    spec = build_phi_dense_plan_spec(
        recipient_config=iq_model.config,
        donor_manifest=donor_runtime.manifest,
        calibration_manifest_hash=calibration_manifest.fingerprint,
        layer_mapping=correspondence.mapping,
        layer_maps=layer_ids,
        embedding_map_id=embedding_map_id,
        final_residual_map_id=final_map_id,
        donor_id=donor_runtime.donor_id,
    )
    plan = build_phi_dense_transport_plan(spec)
    execution = execute_transport_plan(
        iq_model,
        plan,
        [donor_runtime],
        maps,
        training_phase=training_phase,
        apply=True,
    )

    dora_paths = _dora_paths(plan)
    install_dora(
        iq_model,
        dora_paths,
        rank=dora_rank,
        alpha=dora_alpha,
        dropout=0.0,
        freeze_base=True,
    )
    provenance = _final_parameter_provenance(
        iq_model,
        plan,
        execution,
        dora_paths,
        training_phase=training_phase,
    )

    map_metrics = {}
    for map_id, coordinate_map in maps.items():
        diagnostic = coordinate_map.diagnostics
        map_metrics[map_id] = (
            None
            if diagnostic is None
            else {
                "fit_rmse": _json_float(diagnostic.fit_rmse),
                "validation_rmse": _json_float(diagnostic.validation_rmse),
                "effective_rank": diagnostic.effective_rank,
                "condition_number": _json_float(diagnostic.condition_number),
            }
        )
    report = {
        "schema_version": 1,
        "plan_fingerprint": plan.fingerprint,
        "donor_fingerprint": donor_runtime.manifest.fingerprint,
        "calibration_fingerprint": calibration_manifest.fingerprint,
        "iq_config_fingerprint": iq_model.config.fingerprint,
        "layer_mapping": {
            str(k): int(v) for k, v in sorted(correspondence.mapping.items())
        },
        "dora_paths": list(dora_paths),
        "gqa_explained_variance_ratio": gqa_variance,
        "map_metrics": map_metrics,
        "applied_parameters": list(execution.applied_parameters),
    }

    output = Path(output_dir)
    _write_bundle(
        output,
        model=iq_model,
        config=iq_model.config,
        donor_manifest=donor_runtime.manifest,
        calibration_manifest=calibration_manifest,
        plan=plan,
        maps=maps,
        provenance=provenance,
        dora_paths=dora_paths,
        dora_rank=dora_rank,
        dora_alpha=dora_alpha,
        report=report,
    )
    return PhiTransferJobResult(
        output_dir=output,
        donor_manifest=donor_runtime.manifest,
        transport_plan=plan,
        layer_mapping=correspondence.mapping,
        dora_paths=dora_paths,
        execution_report=execution,
        provenance=provenance,
    )


def run_phi_dense_transfer(
    *,
    checkpoint: str | Path,
    recipient_config_path: str | Path,
    fit_batches_path: str | Path,
    validation_batches_path: str | Path,
    calibration_manifest_path: str | Path,
    output_dir: str | Path,
    checkpoint_revision: str,
    donor_license: str,
    source_uri: str | None = None,
    device: str = "cpu",
    dtype: str = "auto",
    dora_rank: int = 64,
    dora_alpha: float | None = None,
    ridge: float = 1e-3,
    shadow_measurements: int = 256,
    shadow_seed: int = 0,
) -> PhiTransferJobResult:
    from .checkpoint import SafetensorsSource

    checkpoint = Path(checkpoint)
    source = SafetensorsSource(checkpoint)
    donor_config = DonorConfig.from_json(checkpoint / "config.json")
    inspector = Phi4Inspector(donor_config)
    inspector.validate_checkpoint(source).require_ok()

    tokenizer_hash = hash_tokenizer_files(checkpoint)
    manifest = build_donor_manifest(
        donor_config,
        source,
        donor_id="phi",
        checkpoint_revision=checkpoint_revision,
        tokenizer_hash=tokenizer_hash,
        license=donor_license,
        operator_layout_version="phi3-fused-v1",
        source_uri=source_uri or checkpoint.resolve().as_uri(),
    )
    runtime = DonorRuntime("phi", manifest, inspector, source)
    calibration = CalibrationManifest.from_json(calibration_manifest_path)
    config = IQModelConfig.from_json(str(recipient_config_path))
    fit_batches = load_token_batches(fit_batches_path).batches
    validation_batches = load_token_batches(validation_batches_path).batches

    phi_model = load_local_phi_causal_lm(
        checkpoint,
        device=device,
        dtype=dtype,
    )
    iq_model = IQForCausalLM(config).to(torch.device(device))
    first_phi_parameter = next(phi_model.parameters())
    if first_phi_parameter.is_floating_point():
        iq_model = iq_model.to(dtype=first_phi_parameter.dtype)

    return run_phi_dense_transfer_loaded(
        phi_model=phi_model,
        iq_model=iq_model,
        donor_runtime=runtime,
        calibration_manifest=calibration,
        fit_batches=fit_batches,
        validation_batches=validation_batches,
        output_dir=output_dir,
        dora_rank=dora_rank,
        dora_alpha=dora_alpha,
        ridge=ridge,
        shadow_measurements=shadow_measurements,
        shadow_seed=shadow_seed,
    )
