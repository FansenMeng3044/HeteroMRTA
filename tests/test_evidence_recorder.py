import json
import tempfile
import unittest
from pathlib import Path

from utils.evidence_recorder import (
    EvidenceRecorder,
    EpisodeEvidenceBuffer,
    select_candidate_action_indices,
)


def make_decision(chosen_match=0.0, alternative_match=1.0):
    return {
        'decision_id': 0,
        'episode_id': 10,
        'current_agent_id': 1,
        'current_agent_species': 0,
        'current_agent_skill_vector': [1, 0],
        'current_open_task_count': 2,
        'current_completed_task_count': 0,
        'remaining_task_count': 2,
        'chosen_task_id': 0,
        'chosen_task_state': 'open',
        'model_entropy': 0.5,
        'valid_action_count': 3,
        'eventual_episode_success': False,
        'eventual_deadlock': True,
        'eventual_makespan': 10.0,
        'eventual_awt': 2.0,
        'eventual_awar': None,
        'candidates': [
            {
                'action_index': 1,
                'task_id': 0,
                'task_state': 'open',
                'is_valid': True,
                'is_chosen': True,
                'model_logit': 0.2,
                'capability_logit_bias': 0.0,
                'biased_model_logit': 0.2,
                'model_prob': 0.6,
                'remaining_requirement': [1, 0],
                'original_requirement': [1, 0],
                'capability_match': chosen_match,
            },
            {
                'action_index': 2,
                'task_id': 1,
                'task_state': 'open',
                'is_valid': True,
                'is_chosen': False,
                'model_logit': 0.1,
                'capability_logit_bias': 0.05,
                'biased_model_logit': 0.15,
                'model_prob': 0.4,
                'remaining_requirement': [0, 1],
                'original_requirement': [0, 1],
                'capability_match': alternative_match,
            },
        ],
        'decoder_logit_debug': {
            'bias_global_step': 3000,
            'bias_apply': True,
            'used_weight': 0.5,
            'used_lambda': 0.1,
            'clip_range': [-2.0, 2.0],
            'action_index_order': '0=depot, action_i=task_id_i_minus_1',
            'valid_action_mask': [True, True, True],
            'capability_match': [0.0, chosen_match, alternative_match],
            'raw_decoder_logits': [0.0, 0.2, 0.1],
            'capability_logit_bias': [0.0, 0.0, 0.05],
            'biased_decoder_logits': [0.0, 0.2, 0.15],
        },
    }


def make_episode(episode_id=10, success=False, deadlock=True):
    return {
        'episode_id': episode_id,
        'success': success,
        'deadlock': deadlock,
        'timeout': False,
        'final_makespan': 10.0,
        'average_waiting_time': 2.0,
        'average_wasted_ability_ratio': None,
        'reward': -11.0,
        'num_agents': 2,
        'num_tasks': 2,
        'num_species': 2,
        'num_skills': 2,
        'task_to_agent_ratio': 1.0,
        'max_open_tasks': 2,
        'average_open_tasks': 1.5,
        'deadlock_step': 2 if deadlock else None,
        'num_decisions': 2,
    }


class CandidateSelectionTest(unittest.TestCase):
    def test_selection_keeps_chosen_top_and_better_match_with_cap(self):
        selected = select_candidate_action_indices(
            probabilities=[0.05, 0.55, 0.25, 0.15],
            valid_actions=[True, True, True, True],
            capability_matches=[None, 0.0, 1.0, 1.0],
            chosen_action=1,
            max_candidates=3,
            top_k=1)
        self.assertEqual([1, 2, 3], selected)

    def test_depot_can_be_selected_without_capability_match(self):
        selected = select_candidate_action_indices(
            probabilities=[0.9, 0.1],
            valid_actions=[True, True],
            capability_matches=[None, 1.0],
            chosen_action=0,
            max_candidates=2,
            top_k=1)
        self.assertEqual([0, 1], selected)


class EpisodeEvidenceBufferTest(unittest.TestCase):
    def test_eventual_outcome_is_backfilled_and_serializable(self):
        buffer = EpisodeEvidenceBuffer(12)
        buffer.record_decision(make_decision())
        payload = buffer.record_episode_end(make_episode(12))
        json.dumps(payload, allow_nan=False)
        decision = payload['decisions'][0]
        self.assertTrue(decision['eventual_deadlock'])
        self.assertEqual(10.0, decision['eventual_makespan'])


class EvidenceRecorderTest(unittest.TestCase):
    def test_default_interval_uses_3000_step_window_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EvidenceRecorder(output_dir=temp_dir)
            self.assertEqual(3000, recorder.interval_steps)
            recorder.record_episode_payload({
                'episode': make_episode(),
                'decisions': [make_decision() for _ in range(3000)],
            })
            self.assertTrue((
                Path(temp_dir)
                / 'evidence_window_00000000_00003000.md'
            ).exists())
            self.assertEqual(
                Path(temp_dir)
                / 'evidence_window_00000000_00003000.md',
                recorder.latest_markdown_report_path)

    def test_complete_window_writes_all_artifacts_and_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EvidenceRecorder(
                interval_steps=2,
                output_dir=temp_dir,
                max_cases_per_report=2)
            payload = {
                'episode': make_episode(),
                'decisions': [make_decision(), make_decision()],
            }
            recorder.record_episode_payload(payload)

            paths = sorted(Path(temp_dir).iterdir())
            self.assertEqual(4, len(paths))
            names = [path.name for path in paths]
            self.assertIn('decisions_00000000_00000002.jsonl', names)
            self.assertIn('episodes_00000000_00000002.jsonl', names)
            self.assertIn('evidence_window_00000000_00000002.json', names)
            self.assertIn('evidence_window_00000000_00000002.md', names)

            report_path = Path(temp_dir) / 'evidence_window_00000000_00000002.json'
            report = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual(
                2,
                report['failure_mode_distribution'][
                    'missed_better_capability_alternative']['count'])
            self.assertEqual(
                {'capability_match': 0.0},
                report['recommended_llm_output_format']['weights'])
            failed_case = report['representative_failed_decision_cases'][0]
            self.assertIn('decoder_logit_debug', failed_case)
            self.assertEqual(
                [0.0, 0.2, 0.1],
                failed_case['decoder_logit_debug']['raw_decoder_logits'])
            self.assertEqual(
                0.05,
                failed_case['candidates'][1]['capability_logit_bias'])

            markdown = (
                Path(temp_dir) / 'evidence_window_00000000_00000002.md'
            ).read_text(encoding='utf-8')
            for section in 'ABCDEFGH':
                self.assertIn('## {}.'.format(section), markdown)
            self.assertIn('Raw decoder logits', markdown)
            self.assertIn('Capability logit bias', markdown)
            self.assertIn('Biased decoder logits', markdown)

    def test_partial_window_flush_and_previous_window_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EvidenceRecorder(interval_steps=2, output_dir=temp_dir)
            recorder.record_episode_payload({
                'episode': make_episode(1),
                'decisions': [make_decision(), make_decision()],
            })
            second_episode = make_episode(2, success=True, deadlock=False)
            recorder.record_episode_payload({
                'episode': second_episode,
                'decisions': [make_decision(1.0, 1.0)],
            })
            written = recorder.flush_window_if_needed(force=True)
            self.assertEqual(4, len(written))
            report = json.loads((
                Path(temp_dir) / 'evidence_window_00000002_00000003.json'
            ).read_text(encoding='utf-8'))
            self.assertIsNotNone(
                report['change_from_previous_window']['success_rate'])

    def test_disabled_recorder_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'disabled'
            recorder = EvidenceRecorder(enabled=False, output_dir=output)
            recorder.record_episode_payload({
                'episode': make_episode(),
                'decisions': [make_decision()],
            })
            recorder.flush_window_if_needed(force=True)
            self.assertFalse(output.exists())

    def test_zero_decision_episode_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EvidenceRecorder(interval_steps=2, output_dir=temp_dir)
            recorder.record_episode_payload({
                'episode': make_episode(3),
                'decisions': [],
            })
            recorder.flush_window_if_needed(force=True)
            path = Path(temp_dir) / 'episodes_00000000_00000000.jsonl'
            self.assertTrue(path.exists())
            record = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(3, record['episode_id'])

    def test_existing_window_restores_step_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = EvidenceRecorder(interval_steps=2, output_dir=temp_dir)
            first.record_episode_payload({
                'episode': make_episode(1),
                'decisions': [make_decision(), make_decision()],
            })

            resumed = EvidenceRecorder(interval_steps=2, output_dir=temp_dir)
            self.assertEqual(2, resumed.global_step)
            self.assertEqual(2, resumed.window_start)
            resumed.record_episode_payload({
                'episode': make_episode(2),
                'decisions': [make_decision()],
            })
            resumed.flush_window_if_needed(force=True)

            self.assertTrue((
                Path(temp_dir) / 'evidence_window_00000002_00000003.json'
            ).exists())
            self.assertTrue((
                Path(temp_dir) / 'evidence_window_00000000_00000002.json'
            ).exists())

    def test_partial_window_resume_keeps_fixed_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = EvidenceRecorder(interval_steps=2, output_dir=temp_dir)
            first.record_episode_payload({
                'episode': make_episode(1),
                'decisions': [make_decision()],
            })
            first.flush_window_if_needed(force=True)

            resumed = EvidenceRecorder(interval_steps=2, output_dir=temp_dir)
            self.assertEqual(1, resumed.global_step)
            self.assertEqual(0, resumed.window_start)
            self.assertEqual(1, len(resumed.decisions))
            resumed.record_episode_payload({
                'episode': make_episode(2),
                'decisions': [make_decision()],
            })

            full_path = (
                Path(temp_dir) / 'evidence_window_00000000_00000002.json')
            self.assertTrue(full_path.exists())
            report = json.loads(full_path.read_text(encoding='utf-8'))
            self.assertEqual(2, report['window']['end_step_inclusive'])
            self.assertEqual(2, report['global_training_trend'][
                'decision_count'])


if __name__ == '__main__':
    unittest.main()
