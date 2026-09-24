from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import torch
from torch import nn

from iq_model import IQModelConfig, MTPConfig, MultiTokenPrediction


class PretrainingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PretrainingObjectiveConfig:
    mtp_loss_weight: float = 0.0
    moe_load_balance_loss_weight: float = 0.0
    moe_router_z_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        weights = {
            "mtp_loss_weight": self.mtp_loss_weight,
            "moe_load_balance_loss_weight": self.moe_load_balance_loss_weight,
            "moe_router_z_loss_weight": self.moe_router_z_loss_weight,
        }
        bad = [name for name, value in weights.items() if float(value) < 0.0]
        if bad:
            raise PretrainingConfigError(
                f"pretraining objective weights must be non-negative: {', '.join(bad)}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PretrainingOutput:
    loss: torch.Tensor | None
    ntp_loss: torch.Tensor | None
    mtp_loss: torch.Tensor | None
    moe_load_balance_loss: torch.Tensor | None
    moe_router_z_loss: torch.Tensor | None
    logits: torch.Tensor


def _language_model_config(main_model: nn.Module) -> IQModelConfig:
    config = getattr(main_model, "model_config", None)
    if isinstance(config, IQModelConfig):
        return config
    config = getattr(main_model, "config", None)
    if isinstance(config, IQModelConfig):
        return config
    raise PretrainingConfigError(
        "main_model must expose IQModelConfig as .model_config or .config"
    )


def _model_config_payload(main_model: nn.Module) -> dict[str, object]:
    config = getattr(main_model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        raise PretrainingConfigError(
            "main_model config must provide deterministic to_dict()"
        )
    payload = config.to_dict()
    if not isinstance(payload, dict):
        raise PretrainingConfigError("main_model config to_dict() must return a dict")
    return payload


class IQPretrainingModel(nn.Module):
    """Training wrapper for dense-control or heterogeneous-hybrid IQ models.

    MTP embedding/output weights are physically shared with the main model.
    MoE auxiliary losses remain owned by the main hybrid backbone but their
    coefficients live here so the model architecture never hardcodes training
    objective weights.
    """

    def __init__(
        self,
        main_model: nn.Module,
        objective_config: PretrainingObjectiveConfig,
        *,
        mtp_config: MTPConfig | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(main_model, "embed_tokens") or not hasattr(main_model, "lm_head"):
            raise PretrainingConfigError(
                "main_model must expose embed_tokens and lm_head"
            )
        if objective_config.mtp_loss_weight > 0 and mtp_config is None:
            raise PretrainingConfigError(
                "mtp_config is required when mtp_loss_weight is positive"
            )

        language_config = _language_model_config(main_model)
        self.main_model = main_model
        self.language_config = language_config
        self.objective_config = objective_config
        self.mtp_config = mtp_config
        self.mtp = (
            MultiTokenPrediction(
                language_config,
                mtp_config,
                shared_embedding=main_model.embed_tokens,
                shared_head=main_model.lm_head,
            )
            if mtp_config is not None
            else None
        )

    @property
    def config(self):
        return self.main_model.config

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": 2,
            "model_config": _model_config_payload(self.main_model),
            "objective_config": self.objective_config.to_dict(),
            "mtp_config": (
                {"num_prediction_layers": self.mtp_config.num_prediction_layers}
                if self.mtp_config is not None
                else None
            ),
        }
        return sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _weighted_auxiliary(
        *,
        name: str,
        value: torch.Tensor | None,
        weight: float,
    ) -> torch.Tensor | None:
        if weight == 0.0:
            return None
        if value is None:
            raise PretrainingConfigError(
                f"{name} weight is positive but the main model did not produce {name}"
            )
        if not bool(torch.isfinite(value)):
            raise RuntimeError(f"{name} is non-finite")
        return float(weight) * value

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> PretrainingOutput:
        use_mtp = (
            self.mtp is not None
            and self.objective_config.mtp_loss_weight > 0
        )
        main_output = self.main_model(
            input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_hidden_states=use_mtp,
        )
        ntp_loss = main_output.loss
        mtp_loss: torch.Tensor | None = None

        if use_mtp:
            if labels is None:
                raise ValueError(
                    "labels are required when MTP training is enabled"
                )
            if main_output.hidden_states is None:
                raise RuntimeError(
                    "main model did not return hidden states required by MTP"
                )
            assert self.mtp is not None
            mtp_output = self.mtp(
                input_ids,
                main_output.hidden_states,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=document_ids,
            )
            mtp_loss = mtp_output.loss
            if mtp_loss is None:
                raise RuntimeError("MTP training did not produce a loss")

        moe_load_balance_loss = getattr(
            main_output,
            "load_balance_loss",
            None,
        )
        moe_router_z_loss = getattr(
            main_output,
            "router_z_loss",
            None,
        )

        terms: list[torch.Tensor] = []
        if ntp_loss is not None:
            terms.append(ntp_loss)
        weighted_mtp = self._weighted_auxiliary(
            name="mtp_loss",
            value=mtp_loss,
            weight=self.objective_config.mtp_loss_weight,
        )
        if weighted_mtp is not None:
            terms.append(weighted_mtp)
        weighted_balance = self._weighted_auxiliary(
            name="load_balance_loss",
            value=moe_load_balance_loss,
            weight=self.objective_config.moe_load_balance_loss_weight,
        )
        if weighted_balance is not None:
            terms.append(weighted_balance)
        weighted_router = self._weighted_auxiliary(
            name="router_z_loss",
            value=moe_router_z_loss,
            weight=self.objective_config.moe_router_z_loss_weight,
        )
        if weighted_router is not None:
            terms.append(weighted_router)

        loss = torch.stack(terms).sum() if terms else None
        return PretrainingOutput(
            loss=loss,
            ntp_loss=ntp_loss,
            mtp_loss=mtp_loss,
            moe_load_balance_loss=moe_load_balance_loss,
            moe_router_z_loss=moe_router_z_loss,
            logits=main_output.logits,
        )
