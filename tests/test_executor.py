from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from iq_model import IQForCausalLM, IQModelConfig
from iq_transfer import (
    CoordinateMap,
    DonorRuntime,
    ExecutionError,
    MappingTensorSource,
    Phi4Inspector,
    TargetAssignment,
    TargetSlot,
    TransportPlan,
    TransferMethod,
    build_donor_manifest,
    execute_transport_plan,
)


class TransportExecutorTests(unittest.TestCase):
    def source_and_runtime(self):
        config = {
            "_name_or_path": "test-phi4",
            "model_type": "phi3",
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "torch_dtype": "float32",
        }
        rng = np.random.default_rng(12)
        tensors = {
            "model.layers.0.self_attn.qkv_proj.weight": rng.normal(size=(16, 8)),
            "model.layers.0.self_attn.o_proj.weight": rng.normal(size=(8, 8)),
            "model.layers.0.mlp.gate_up_proj.weight": rng.normal(size=(24, 8)),
            "model.layers.0.mlp.down_proj.weight": rng.normal(size=(8, 12)),
            "model.embed_tokens.weight": rng.normal(size=(32, 8)),
            "lm_head.weight": rng.normal(size=(32, 8)),
        }
        source = MappingTensorSource(tensors)
        inspector = Phi4Inspector.from_config_mapping(config)
        manifest = build_donor_manifest(
            inspector.config,
            source,
            donor_id="phi",
            checkpoint_revision="test-revision",
            tokenizer_hash="tokenizer-hash",
            license="test-license",
            operator_layout_version="phi3-fused-v1",
            source_uri="memory://phi",
            allow_metadata_only=True,
        )
        return source, inspector, manifest

    def target(self):
        config = IQModelConfig(
            vocab_size=32,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=6,
            max_position_embeddings=16,
            rope_theta=10000.0,
        )
        return IQForCausalLM(config)

    def plan_and_maps(self, model, manifest):
        rng = np.random.default_rng(13)
        p_res = rng.normal(size=(8, 4))
        p_q = rng.normal(size=(8, 4))
        maps = {
            "resid": CoordinateMap(p_res, 1e-3, "phi.residual", "iq.residual"),
            "q": CoordinateMap(p_q, 1e-3, "phi.q", "iq.q"),
        }
        assignments = (
            TargetAssignment(
                target_module_path="embed_tokens",
                target_slot=TargetSlot.EMBEDDING,
                target_shape=(32, 4),
                transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                source_donor_id="phi",
                source_operator="model.embed_tokens.weight",
                output_map_id="resid",
            ),
            TargetAssignment(
                target_module_path="lm_head",
                target_slot=TargetSlot.LM_HEAD,
                target_shape=(32, 4),
                transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                source_donor_id="phi",
                source_operator="lm_head.weight",
                input_map_id="resid",
            ),
            TargetAssignment(
                target_module_path="blocks.0.attn.q_proj",
                target_slot=TargetSlot.ATTN_Q,
                target_shape=(4, 4),
                transfer_method=TransferMethod.OPERATOR_TRANSPORT,
                source_donor_id="phi",
                source_layer=0,
                source_operator="attn.q",
                input_map_id="resid",
                output_map_id="q",
            ),
            TargetAssignment(
                target_module_path="norm",
                target_slot=TargetSlot.NORM_SCALE,
                target_shape=(4,),
                transfer_method=TransferMethod.RECIPIENT_NATIVE,
            ),
        )
        plan = TransportPlan(
            plan_id="phi-dense-proof",
            donor_fingerprints=(("phi", manifest.fingerprint),),
            assignments=assignments,
            calibration_manifest_hash="calibration-hash",
            iq_config_hash=model.config.fingerprint,
        )
        return plan, maps

    def test_executor_transports_lexical_and_attention_weights_into_real_model(self):
        source, inspector, manifest = self.source_and_runtime()
        model = self.target()
        plan, maps = self.plan_and_maps(model, manifest)
        runtime = DonorRuntime("phi", manifest, inspector, source)

        dry = execute_transport_plan(model, plan, [runtime], maps, apply=False)
        self.assertEqual(dry.applied_parameters, ())
        values = {update.target_parameter: np.asarray(update.value) for update in dry.updates}

        e_source = source.get("model.embed_tokens.weight")
        lm_source = source.get("lm_head.weight")
        q_source = inspector.operators(source)[0].materialize(source)
        p_res = maps["resid"].matrix
        p_q = maps["q"].matrix

        self.assertTrue(np.allclose(values["embed_tokens.weight"], e_source @ p_res))
        self.assertTrue(np.allclose(values["lm_head.weight"], lm_source @ np.linalg.pinv(p_res).T))
        self.assertTrue(
            np.allclose(
                values["blocks.0.attn.q_proj.weight"],
                p_q.T @ q_source @ np.linalg.pinv(p_res).T,
            )
        )
        self.assertEqual(dry.recipient_native, ("norm.weight",))
        self.assertEqual(len(dry.provenance.records()), 3)

        report = execute_transport_plan(model, plan, [runtime], maps, apply=True)
        self.assertEqual(
            set(report.applied_parameters),
            {"embed_tokens.weight", "lm_head.weight", "blocks.0.attn.q_proj.weight"},
        )
        self.assertTrue(
            np.allclose(
                model.blocks[0].attn.q_proj.weight.detach().cpu().numpy(),
                values["blocks.0.attn.q_proj.weight"],
                atol=1e-6,
            )
        )

    def test_executor_rejects_donor_fingerprint_mismatch_before_mutation(self):
        source, inspector, manifest = self.source_and_runtime()
        model = self.target()
        plan, maps = self.plan_and_maps(model, manifest)
        bad_manifest = replace(manifest, checkpoint_revision="different")
        runtime = DonorRuntime("phi", bad_manifest, inspector, source)
        before = model.blocks[0].attn.q_proj.weight.detach().clone()
        with self.assertRaises(ExecutionError):
            execute_transport_plan(model, plan, [runtime], maps, apply=True)
        self.assertTrue(
            np.array_equal(
                before.cpu().numpy(),
                model.blocks[0].attn.q_proj.weight.detach().cpu().numpy(),
            )
        )


if __name__ == "__main__":
    unittest.main()
