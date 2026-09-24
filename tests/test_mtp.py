from __future__ import annotations

import unittest

import torch

from iq_model import (
    IQForCausalLM,
    IQModelConfig,
    MTPConfig,
    MTPConfigError,
    MultiTokenPrediction,
)


class MultiTokenPredictionTests(unittest.TestCase):
    def config(self) -> IQModelConfig:
        return IQModelConfig(
            vocab_size=41,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )

    def build(self, depth: int = 2):
        config = self.config()
        main = IQForCausalLM(config)
        mtp = MultiTokenPrediction(
            config,
            MTPConfig(num_prediction_layers=depth),
            shared_embedding=main.embed_tokens,
            shared_head=main.lm_head,
        )
        return main, mtp

    def test_sequential_depth_shapes_shared_weights_and_gradients(self):
        torch.manual_seed(21)
        main, mtp = self.build(depth=2)
        self.assertIs(mtp.embed_tokens, main.embed_tokens)
        self.assertIs(mtp.shared_head, main.lm_head)

        ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
            ]
        )
        hidden = main(ids, return_hidden_states=True).hidden_states
        output = mtp(ids, hidden, labels=ids)

        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(len(output.depth_outputs), 2)
        self.assertEqual(tuple(output.depth_outputs[0].logits.shape), (2, 5, 41))
        self.assertEqual(tuple(output.depth_outputs[1].logits.shape), (2, 4, 41))
        self.assertEqual(output.depth_outputs[0].valid_target_count, 8)
        self.assertEqual(output.depth_outputs[1].valid_target_count, 6)

        output.loss.backward()
        self.assertIsNotNone(main.embed_tokens.weight.grad)
        self.assertTrue(torch.isfinite(main.embed_tokens.weight.grad).all())
        self.assertIsNotNone(main.lm_head.weight.grad)
        self.assertTrue(torch.isfinite(main.lm_head.weight.grad).all())
        for layer in mtp.layers:
            for parameter in layer.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_packed_documents_do_not_form_cross_document_mtp_chains(self):
        torch.manual_seed(22)
        main, mtp = self.build(depth=1)
        docs = torch.tensor([[0, 0, 0, 1, 1, 1]])
        ids_a = torch.tensor([[1, 2, 3, 10, 11, 12]])
        ids_b = torch.tensor([[7, 8, 9, 10, 11, 12]])

        hidden_a = main(
            ids_a,
            document_ids=docs,
            return_hidden_states=True,
        ).hidden_states
        hidden_b = main(
            ids_b,
            document_ids=docs,
            return_hidden_states=True,
        ).hidden_states

        out_a = mtp(
            ids_a,
            hidden_a,
            labels=ids_a,
            document_ids=docs,
        )
        out_b = mtp(
            ids_b,
            hidden_b,
            labels=ids_b,
            document_ids=docs,
        )
        self.assertEqual(out_a.depth_outputs[0].valid_target_count, 2)
        self.assertEqual(out_b.depth_outputs[0].valid_target_count, 2)

        # This MTP state uses only document 1 context and predicts the final
        # document-1 token; edits to document 0 must not change it.
        self.assertTrue(
            torch.allclose(
                out_a.depth_outputs[0].logits[:, 3],
                out_b.depth_outputs[0].logits[:, 3],
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_no_labels_returns_logits_without_training_loss(self):
        torch.manual_seed(23)
        main, mtp = self.build(depth=1)
        ids = torch.tensor([[1, 2, 3, 4]])
        hidden = main(ids, return_hidden_states=True).hidden_states
        output = mtp(ids, hidden)
        self.assertIsNone(output.loss)
        self.assertIsNone(output.depth_outputs[0].loss)
        self.assertEqual(tuple(output.depth_outputs[0].logits.shape), (1, 3, 41))

    def test_depth_and_config_validation(self):
        with self.assertRaises(MTPConfigError):
            MTPConfig(num_prediction_layers=0)

        main, mtp = self.build(depth=2)
        ids = torch.tensor([[1, 2, 3]])
        hidden = main(ids, return_hidden_states=True).hidden_states
        with self.assertRaises(ValueError):
            mtp(ids, hidden, labels=ids)


if __name__ == "__main__":
    unittest.main()
