"""GLM-5.3 pinned 78-block topology and actual DSA sharing correctness tests."""
from __future__ import annotations

import unittest
import torch

from iq_transfer.glm53_parallel_contract import (
    GLM53Topology, GLM53TopologyError,
    expected_indexer_types, index_source_layers,
)
from iq_model import CompressedSparseContextAttention
from iq_model.attention.compressed import CompressedContextError, DeepseekV4CSACache
from test_parallel_cached_context import tiny_context


def published_fields() -> dict:
    return {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "indexer_types": list(expected_indexer_types()),
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "num_nextn_predict_layers": 1,
    }


class GLM53TopologyContractTests(unittest.TestCase):
    def test_pinned_78_decoder_blocks_expand_to_156_megatron_physical_layers(self):
        topology = GLM53Topology.from_hf_config(published_fields())
        self.assertEqual(topology.megatron_physical_layers, 156)
        self.assertEqual(topology.dense_blocks, (0, 1, 2))
        self.assertEqual(len(topology.sparse_blocks), 75)
        self.assertEqual(topology.num_routed_experts, 256)
        self.assertEqual(topology.top_k, 8)
        self.assertEqual(topology.index_source_blocks[3:7], (2, 2, 2, 6))
        self.assertEqual(topology.index_source_blocks[-1], 74)
        self.assertEqual(len(topology.export_manifest()["stages"]), 78)
        self.assertEqual(topology.export_manifest()["stages"][77]["megatron_feedforward_physical"], 155)
        self.assertEqual(topology.export_manifest(), topology.export_manifest())

    def test_rejects_modified_layout_and_fake_full_indexer(self):
        spec = published_fields()
        spec["n_routed_experts"] = 32
        with self.assertRaisesRegex(GLM53TopologyError, "n_routed_experts"):
            GLM53Topology.from_hf_config(spec)
        spec = published_fields()
        spec["indexer_types"][6] = "shared"
        with self.assertRaisesRegex(GLM53TopologyError, "IndexShare"):
            GLM53Topology.from_hf_config(spec)
        spec = published_fields()
        spec["mlp_layer_types"][2] = "sparse"
        with self.assertRaisesRegex(GLM53TopologyError, "dense/MoE"):
            GLM53Topology.from_hf_config(spec)
        with self.assertRaisesRegex(GLM53TopologyError, "78"):
            index_source_layers(("shared",) * 2)


class IndexShareAttentionTests(unittest.TestCase):
    def test_shared_topk_full_sequence_equals_prefill_and_step(self):
        torch.manual_seed(209)
        config = tiny_context(direct=True)
        producer = CompressedSparseContextAttention(config).eval()
        consumer = CompressedSparseContextAttention(config).eval()
        producer_x = torch.randn(2, 8, 16)
        consumer_x = torch.randn(2, 8, 16)
        with torch.no_grad():
            full_indices: list[torch.Tensor] = []
            producer(producer_x, capture_selection=full_indices)
            self.assertEqual(len(full_indices), 1)
            self.assertEqual(full_indices[0].shape[:2], (2, 8))
            full = consumer(consumer_x, shared_selection=full_indices[0])

            p_cache = DeepseekV4CSACache(config)
            c_cache = DeepseekV4CSACache(config)
            pieces = []
            for start, end in ((0, 5), (5, 6), (6, 7), (7, 8)):
                this_index: list[torch.Tensor] = []
                producer(producer_x[:, start:end], past_key_values=p_cache,
                         capture_selection=this_index)
                pieces.append(consumer(
                    consumer_x[:, start:end], past_key_values=c_cache,
                    shared_selection=this_index[0],
                ))
        torch.testing.assert_close(
            torch.cat(pieces, dim=1), full, atol=3e-5, rtol=2e-4
        )

    def test_shared_indexer_fails_closed_on_future_and_missing_document_boundaries(self):
        config = tiny_context(direct=True)
        shared = CompressedSparseContextAttention(config).eval()
        x = torch.randn(1, 4, 16)
        with self.assertRaisesRegex(CompressedContextError, "future"):
            shared(x, shared_selection=torch.tensor([[[3], [3], [3], [3]]]))
        with self.assertRaisesRegex(CompressedContextError, "single unpadded document"):
            shared(x, document_ids=torch.zeros(1, 4, dtype=torch.long),
                   shared_selection=torch.zeros(1, 4, 1, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
