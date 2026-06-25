import unittest

import torch

from attention import AttentionNet


class AttentionEvidenceOutputTest(unittest.TestCase):
    def test_optional_logits_preserve_default_probabilities(self):
        torch.manual_seed(3)
        network = AttentionNet(agent_input_dim=4, task_input_dim=5, embedding_dim=16)
        tasks = torch.rand(1, 4, 5) + 0.1
        agents = torch.rand(1, 3, 4) + 0.1
        mask = torch.tensor([[False, False, True, False]])
        index = torch.tensor([[[1]]])

        default_probs, default_logps = network(tasks, agents, mask, index)
        detailed_probs, detailed_logps, logits = network(
            tasks, agents, mask, index, return_details=True)

        self.assertTrue(torch.allclose(default_probs, detailed_probs))
        self.assertTrue(torch.allclose(default_logps, detailed_logps))
        self.assertTrue(torch.allclose(
            torch.softmax(logits, dim=-1), detailed_probs))

    def test_all_details_separates_raw_and_biased_logits(self):
        torch.manual_seed(7)
        network = AttentionNet(
            agent_input_dim=4, task_input_dim=5, embedding_dim=16)
        tasks = torch.rand(1, 4, 5) + 0.1
        agents = torch.rand(1, 3, 4) + 0.1
        mask = torch.tensor([[False, False, True, False]])
        index = torch.tensor([[[1]]])
        explicit_features = torch.tensor([[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]])
        bias_params = torch.tensor([[
            1.0, 0.5, -2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
        ]])

        (probs, _, final_logits, raw_logits, logit_bias,
         biased_logits) = network(
            tasks,
            agents,
            mask,
            index,
            return_details='all',
            explicit_features=explicit_features,
            bias_params=bias_params)

        torch.testing.assert_close(logit_bias[0], torch.tensor(
            [0.0, 0.5, 0.0, 0.0]))
        self.assertAlmostEqual(
            0.5,
            logit_bias[0, 1].item(),
            places=6)
        self.assertAlmostEqual(
            0.0,
            (biased_logits[0, 0] - raw_logits[0, 0]).item(),
            places=6)
        self.assertAlmostEqual(
            0.5,
            (biased_logits[0, 1] - raw_logits[0, 1]).item(),
            places=6)
        self.assertEqual(0.0, probs[0, 2].item())


if __name__ == '__main__':
    unittest.main()
