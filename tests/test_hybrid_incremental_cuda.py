from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from iq_model import (
    CompressedContextConfig,
    BlockAttnResConfig,
    StableLatentMoEConfig,
    HybridSchedule,
    IQHybridConfig,
    IQHybridForCausalLM,
    IQModelConfig,
    Mamba3MIMOConfig,
    RoutedMoEConfig,
    inspect_mamba3_mimo_runtime,
)


class HybridIncrementalCudaTests(unittest.TestCase):
    def test_real_mamba3_mimo_and_csa_hca_dense_prefill_decode(self):
        # This tests actual pinned Mamba-3 CUDA kernels, never a fake mixer.
        runtime = inspect_mamba3_mimo_runtime()
        if not runtime.ready:
            self.skipTest("pinned CUDA Mamba-3 MIMO runtime unavailable")
        from mamba_ssm.modules import mamba3 as mamba3_module

        if mamba3_module.mamba3_step_fn is None:
            self.skipTest("upstream Mamba-3 CuTe incremental step kernel unavailable")

        config = IQHybridConfig(
            model=IQModelConfig(
                vocab_size=97,
                hidden_size=64,
                num_hidden_layers=10,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=128,
                max_position_embeddings=64,
                rope_theta=10000.0,
            ),
            schedule=HybridSchedule.parse("M E M C E M H E M A"),
            mamba3=Mamba3MIMOConfig(
                d_model=64,
                num_layers=4,
                d_state=128,
                headdim=64,
                mimo_rank=4,
                expand=2.0,
                rope_fraction=0.5,
                chunk_size=16,
            ),
            moe=RoutedMoEConfig(
                hidden_size=64,
                expert_intermediate_size=128,
                num_experts=4,
                top_k=2,
                shared_expert_intermediate_size=64,
            ),
            compressed_context=CompressedContextConfig(
                hidden_size=64,
                num_attention_heads=4,
                head_dim=16,
                q_lora_rank=32,
                partial_rotary_dim=8,
                max_position_embeddings=64,
                sliding_window=3,
                csa_compress_rate=2,
                hca_compress_rate=4,
                o_groups=2,
                o_lora_rank=8,
                index_n_heads=2,
                index_head_dim=16,
                index_topk=2,
                compress_rope_theta=10000.0,
            ),
        )
        canonical = replace(
            config,
            moe_variant="stable_latent",
            stable_moe=StableLatentMoEConfig(
                hidden_size=64,
                latent_size=32,
                expert_intermediate_size=64,
                num_experts=4,
                top_k=2,
            ),
            attnres=BlockAttnResConfig(
                hidden_size=64,
                num_layers=len(config.schedule.layers),
                block_size=2,
            ),
        )
        torch.manual_seed(81)
        tokens = torch.randint(0, 97, (1, 11), device="cuda")
        for model_config in (config, canonical):
            with self.subTest(moe_variant=model_config.moe_variant):
                model = IQHybridForCausalLM(
                    model_config, dtype=torch.bfloat16, device="cuda"
                ).eval()
                with torch.no_grad():
                    full = model(tokens).logits
                    for first_chunk in (1, 3, 7, 11):
                        state = model.allocate_inference_cache(
                            batch_size=1, max_seqlen=16
                        )
                        parts = [
                            model(tokens[:, :first_chunk], inference_cache=state).logits
                        ]
                        for index in range(first_chunk, tokens.shape[1]):
                            parts.append(
                                model(
                                    tokens[:, index:index + 1],
                                    inference_cache=state,
                                ).logits
                            )
                        actual = torch.cat(parts, dim=1)
                        torch.testing.assert_close(
                            actual, full, atol=0.05, rtol=0.05
                        )
                        self.assertEqual(state.mamba.seqlen_offset, 11)
                        state.reset()
                        self.assertEqual(state.mamba.seqlen_offset, 0)
                        self.assertFalse(state.mamba.key_value_memory_dict)
                        # The pinned Mamba3.step must be deterministic after
                        # resetting all four recurrent states.
                        replay = [
                            model(
                                tokens[:, :first_chunk], inference_cache=state
                            ).logits
                        ]
                        for index in range(first_chunk, tokens.shape[1]):
                            replay.append(
                                model(
                                    tokens[:, index:index + 1],
                                    inference_cache=state,
                                ).logits
                            )
                        torch.testing.assert_close(
                            torch.cat(replay, dim=1),
                            actual,
                            atol=0.05,
                            rtol=0.05,
                        )


if __name__ == "__main__":
    unittest.main()
