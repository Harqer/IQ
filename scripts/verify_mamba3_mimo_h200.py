from __future__ import annotations

import torch

from iq_model import Mamba3MIMOConfig, Mamba3MIMOState, require_mamba3_mimo_runtime


RTOL = 0.1
ATOL = 0.1
BATCH = 1
SEQLEN = 32


def _assert_h200() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("H200 verification requires CUDA")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if "H200" not in name:
        raise SystemExit(f"H200 verification must run on H200; found {name!r}")
    require_mamba3_mimo_runtime(device)
    return device


def _config() -> Mamba3MIMOConfig:
    # Use IQ's canonical MIMO rank-4 production configuration.
    # Pinned Mamba-3 recommends 64 / rank = 16 for all supported dtypes.
    return Mamba3MIMOConfig.production_4096x32()


def _assert_all_finite_gradients(model: torch.nn.Module) -> None:
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
    if missing or nonfinite:
        raise AssertionError(
            f"Mamba-3 MIMO gradient failure: missing={missing}, nonfinite={nonfinite}"
        )


def main() -> None:
    device = _assert_h200()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    prod = Mamba3MIMOState(
        _config(),
        layer_idx=0,
        dtype=torch.bfloat16,
        device=device,
    ).train()

    x_train = torch.randn(
        BATCH, SEQLEN, 4096, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    y_train = prod(x_train)
    if y_train.shape != x_train.shape or not bool(torch.isfinite(y_train).all()):
        raise AssertionError("Mamba-3 MIMO BF16 forward failed")
    loss = y_train.float().square().mean()
    loss.backward()
    if x_train.grad is None or not bool(torch.isfinite(x_train.grad).all()):
        raise AssertionError("Mamba-3 MIMO input gradient is missing/non-finite")
    _assert_all_finite_gradients(prod)

    prod.eval()
    ref = Mamba3MIMOState(
        _config(),
        layer_idx=0,
        dtype=torch.float32,
        device=device,
    ).eval()
    ref.load_state_dict(
        {name: tensor.detach().float() for name, tensor in prod.state_dict().items()},
        strict=True,
    )

    x = torch.randn(BATCH, SEQLEN, 4096, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        full_ref = ref(x.float())

        state = prod.allocate_inference_cache(
            BATCH, SEQLEN, device=device, dtype=torch.bfloat16
        )
        step_outputs = []
        for t in range(SEQLEN):
            out, angle, ssm, k_state, v_state = prod.core.step(x[:, t], *state)
            state = (angle, ssm, k_state, v_state)
            step_outputs.append(out)
        step_outputs = torch.stack(step_outputs, dim=1)
        torch.testing.assert_close(
            step_outputs.float(), full_ref.float(), rtol=RTOL, atol=ATOL
        )

        split = SEQLEN // 2
        mixed = Mamba3MIMOState(
            _config(),
            layer_idx=0,
            dtype=torch.bfloat16,
            device=device,
        ).eval()
        mixed.load_state_dict(prod.state_dict(), strict=True)
        from mamba_ssm.utils.generation import InferenceParams

        inference = InferenceParams(max_seqlen=SEQLEN, max_batch_size=BATCH)
        prefix = mixed(x[:, :split], inference_params=inference)
        inference.seqlen_offset = split
        suffix = []
        for t in range(split, SEQLEN):
            suffix.append(mixed(x[:, t : t + 1], inference_params=inference))
            inference.seqlen_offset += 1
        mixed_out = torch.cat([prefix, *suffix], dim=1)
        if inference.seqlen_offset != SEQLEN:
            raise AssertionError("Mamba-3 streaming inference offset was not advanced")
        torch.testing.assert_close(
            mixed_out.float(), full_ref.float(), rtol=RTOL, atol=ATOL
        )

    print(
        "Mamba-3 rank-4 MIMO H200 gate passed:",
        {
            "device": torch.cuda.get_device_name(device),
            "shape": list(full_ref.shape),
            "rtol": RTOL,
            "atol": ATOL,
        },
    )


if __name__ == "__main__":
    main()
