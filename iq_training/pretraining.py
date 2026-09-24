from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import torch
from torch import nn

from iq_model import IQForCausalLM, MTPConfig, MultiTokenPrediction


class PretrainingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PretrainingObjectiveConfig:
    mtp_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.mtp_loss_weight < 0:
            raise PretrainingConfigError("mtp_loss_weight must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PretrainingOutput:
    loss: torch.Tensor | None
    ntp_loss: torch.Tensor | None
    mtp_loss: torch.Tensor | None
    logits: torch.Tensor


class IQPretrainingModel(nn.Module):
    """Training-time wrapper for the main model plus optional MTP modules.

    MTP embedding/output weights are physically shared with the main model.
    The wrapper's fingerprint includes the base model, MTP depth, and objective
    weights so checkpoint compatibility covers the complete training graph.
    """

    def __init__(
        self,
        main_model: IQForCausalLM,
        objective_config: PretrainingObjectiveConfig,
        *,
        mtp_config: MTPConfig | None = None,
    ) -> None:
        super().__init__()
        if objective_config.mtp_loss_weight > 0 and mtp_config is None:
            raise PretrainingConfigError(
                "mtp_config is required when mtp_loss_weight is positive"
            )
        self.main_model = main_model
        self.objective_config = objective_config
        self.mtp_config = mtp_config
        self.mtp = (
            MultiTokenPrediction(
                main_model.config,
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
            "schema_version": 1,
            "model_config": self.main_model.config.to_dict(),
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

        loss = ntp_loss
        if mtp_loss is not None:
            weighted_mtp = self.objective_config.mtp_loss_weight * mtp_loss
            loss = weighted_mtp if loss is None else loss + weighted_mtp

        return PretrainingOutput(
            loss=loss,
            ntp_loss=ntp_loss,
            mtp_loss=mtp_loss,
            logits=main_output.logits,
        )
