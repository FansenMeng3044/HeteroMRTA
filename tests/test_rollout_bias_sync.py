import unittest

from utils.rollout_bias_sync import (
    bias_snapshot_version,
    complete_evidence_payloads,
    rollout_uses_active_snapshot,
    slice_transition_buffer,
    truncate_evidence_payloads,
)


class RolloutBiasSyncTest(unittest.TestCase):
    def test_version_match_rejects_late_old_worker_result(self):
        active = {'global_step': 3000}
        self.assertEqual(3000, bias_snapshot_version(active))
        self.assertTrue(rollout_uses_active_snapshot(
            {'bias_global_step': 3000}, active))
        self.assertFalse(rollout_uses_active_snapshot(
            {'bias_global_step': 0}, active))

    def test_buffer_and_evidence_stop_at_exact_window_boundary(self):
        buffer = {key: list(range(7)) for key in range(9)}
        sliced = slice_transition_buffer(buffer, 5)
        self.assertTrue(all(len(values) == 5 for values in sliced.values()))

        payloads = [
            {
                'episode': {'episode_id': 1},
                'decisions': [{'decision_id': index} for index in range(3)],
            },
            {
                'episode': {'episode_id': 2},
                'decisions': [{'decision_id': index} for index in range(4)],
            },
        ]
        truncated, kept = truncate_evidence_payloads(payloads, 5)
        self.assertEqual(5, kept)
        self.assertEqual(2, len(truncated))
        self.assertEqual(3, len(truncated[0]['decisions']))
        self.assertEqual(2, len(truncated[1]['decisions']))

    def test_missing_optional_evidence_is_padded_without_losing_steps(self):
        completed = complete_evidence_payloads([], 3, episode_number=7)
        self.assertEqual(1, len(completed))
        self.assertEqual(3, len(completed[0]['decisions']))
        self.assertTrue(completed[0]['episode']['evidence_missing'])
        self.assertTrue(all(
            decision['evidence_missing']
            for decision in completed[0]['decisions']))


if __name__ == '__main__':
    unittest.main()
