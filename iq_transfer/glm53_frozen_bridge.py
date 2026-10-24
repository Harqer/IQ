"""Frozen GLM-5.3 SwiGLU expert with a trainable low-rank IQ latent interface.

The donor BF16 gate/up/down tensors are immutable registered buffers.
Only small adapter factors receive gradients. The IQ latent basis is frozen.
This is an isolated expert experiment, NOT a deployable IQ checkpoint.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrozenExpertBridgeError(ValueError):
    pass


class FrozenGLMExpertBridge(nn.Module):
    """Learn nonlinear-manifold corrections around a fixed linear IQ codec.

    Input: IQ latent z=[batch, K]. Reconstruct donor input using P.T, then
    add a rank-r learnable residual. Execute ORIGINAL frozen BF16 donor FFN.
    Compress donor output using P, optionally plus learned rank-r residual.
    """

    def __init__(
        self,
        donor: Mapping[str, Tensor],
        basis: Tensor,
        *,
        rank: int = 16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if basis.ndim != 2 or not 1 <= rank <= basis.shape[1]:
            raise FrozenExpertBridgeError("invalid [donor, latent] basis or rank")
        width, latent = map(int, basis.shape)
        if latent >= width or latent < 1:
            raise FrozenExpertBridgeError("latent width must be smaller than donor width")
        required = {"gate", "up", "down"}
        if set(donor) != required:
            raise FrozenExpertBridgeError("donor requires gate, up and down weights")
        intermediate = int(donor["gate"].shape[0])
        expected = {
            "gate": (intermediate, width),
            "up": (intermediate, width),
            "down": (width, intermediate),
        }
        for key in sorted(required):
            tensor = donor[key]
            if (
                not isinstance(tensor, Tensor)
                or tuple(tensor.shape) != expected[key]
                or tensor.dtype != torch.bfloat16
                or not bool(torch.isfinite(tensor).all())
            ):
                raise FrozenExpertBridgeError(f"invalid original BF16 expert {key}")
            self.register_buffer(key, tensor.detach().contiguous().clone())
        if not bool(torch.isfinite(basis).all()):
            raise FrozenExpertBridgeError("nonfinite IQ coordinate basis")
        p = basis.detach().to(torch.float32).contiguous().clone()
        gram = p.T @ p
        if not torch.allclose(
            gram, torch.eye(latent, device=p.device), atol=5e-4, rtol=0
        ):
            raise FrozenExpertBridgeError("IQ coordinate basis must be orthonormal")
        self.register_buffer("basis", p)

        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.input_a = nn.Parameter(
            torch.randn(latent, rank, generator=rng) * 0.005
        )
        self.input_b = nn.Parameter(torch.zeros(rank, width))
        self.output_a = nn.Parameter(
            torch.randn(width, rank, generator=rng) * 0.005
        )
        self.output_b = nn.Parameter(torch.zeros(rank, latent))

    def decode_input(self, latent_input: Tensor) -> Tensor:
        z = latent_input.float()
        return (z @ self.basis.T) + (z @ self.input_a) @ self.input_b

    def encode_output(self, donor_output: Tensor) -> Tensor:
        y = donor_output.float()
        return (y @ self.basis) + (y @ self.output_a) @ self.output_b

    def donor_forward(self, donor_input: Tensor) -> Tensor:
        # Cast ORIGINAL BF16 tensors to float for well-defined CPU math;
        # parameter gradients never flow into the donor buffers.
        x = donor_input.float()
        return F.linear(
            F.silu(F.linear(x, self.gate.float()))
            * F.linear(x, self.up.float()),
            self.down.float(),
        )

    def forward(self, latent_input: Tensor) -> Tensor:
        if latent_input.ndim != 2 or latent_input.shape[1] != self.basis.shape[1]:
            raise FrozenExpertBridgeError("IQ latent input shape mismatch")
        if not bool(torch.isfinite(latent_input).all()):
            raise FrozenExpertBridgeError("nonfinite IQ latent input")
        return self.encode_output(self.donor_forward(self.decode_input(latent_input)))

    def adapter_state(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, name).detach().to("cpu").contiguous().clone()
            for name in ("input_a", "input_b", "output_a", "output_b")
        }

    @property
    def trainable_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def normalized_latent_mse(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise FrozenExpertBridgeError("latent target and prediction shape mismatch")
    if not bool(torch.isfinite(target).all()):
        raise FrozenExpertBridgeError("nonfinite teacher latent targets")
    return (prediction.float() - target.float()).square().mean() / (
        target.float().square().mean().clamp_min(1e-12)
    )


def train_frozen_bridge(
    model: FrozenGLMExpertBridge,
    z_train: Tensor,
    target_train: Tensor,
    z_eval: Tensor,
    target_eval: Tensor,
    *,
    steps: int = 60,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    if (
        not 1 <= steps <= 2000
        or not 1 <= batch_size <= 256
        or not 0 < learning_rate <= 0.1
        or z_train.ndim != 2
        or z_eval.ndim != 2
        or z_train.shape[0] < 8
        or z_eval.shape[0] < 2
        or target_train.shape != z_train
        or target_eval.shape != z_eval
        or z_train.shape[1] != model.basis.shape[1]
        or not all(bool(torch.isfinite(x).all()) for x in (
            z_train, target_train, z_eval, target_eval,
        ))
    ):
        raise FrozenExpertBridgeError(
            "invalid train/eval inputs or insufficient disjoint samples"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def measure() -> float:
        model.eval()
        with torch.no_grad():
            value = normalized_latent_mse(model(z_eval), target_eval)
            return float(value)

    starting_loss = measure()
    starting_train = float(normalized_latent_mse(
        model(z_train), target_train
    ))
    best_loss = starting_loss
    best_state = model.adapter_state()
    model.train()
    for step in range(steps):
        indices = torch.randint(
            z_train.shape[0], (min(batch_size, z_train.shape[0]),),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        result = model(z_train[indices])
        loss = normalized_latent_mse(result, target_train[indices])
        if not bool(torch.isfinite(loss)):
            raise FrozenExpertBridgeError("nonfinite bridge training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == steps - 1 or (step + 1) % 5 == 0:
            current_loss = measure()
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = model.adapter_state()
            model.train()

    with torch.no_grad():
        for name, value in best_state.items():
            getattr(model, name).copy_(value.to(getattr(model, name).device))
    model.eval()
    end_train = float(normalized_latent_mse(
        model(z_train), target_train
    ))
    return {
        "steps": steps,
        "training_samples": int(z_train.shape[0]),
        "evaluation_samples": int(z_eval.shape[0]),
        "train_loss_before": starting_train,
        "train_loss_after": end_train,
        "heldout_loss_before": starting_loss,
        "heldout_loss_best": best_loss,
        "selected_checkpoint": "best_disjoint_document_eval_or_initial",
        "trainable_parameters": model.trainable_count,
        "original_expert_frozen": True,
    }
