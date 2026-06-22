import copy
import json
import random
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch


if 'matplotlib' not in sys.modules:
    matplotlib = types.ModuleType('matplotlib')
    matplotlib.pyplot = types.ModuleType('matplotlib.pyplot')
    matplotlib.patches = types.ModuleType('matplotlib.patches')
    matplotlib.animation = types.ModuleType('matplotlib.animation')
    matplotlib.offsetbox = types.ModuleType('matplotlib.offsetbox')
    matplotlib.animation.FuncAnimation = object
    matplotlib.offsetbox.OffsetImage = object
    matplotlib.offsetbox.AnnotationBbox = object
    sys.modules['matplotlib'] = matplotlib
    sys.modules['matplotlib.pyplot'] = matplotlib.pyplot
    sys.modules['matplotlib.patches'] = matplotlib.patches
    sys.modules['matplotlib.animation'] = matplotlib.animation
    sys.modules['matplotlib.offsetbox'] = matplotlib.offsetbox

from parameters import EvidenceParams
from utils.bias_manager import compute_capability_match_bias
from worker import Worker


class FakeEnv:
    def __init__(self, *args, **kwargs):
        self.tasks_num = 1
        self.agents_num = 1
        self.species_num = 1
        self.traits_dim = 5
        self.current_time = 0.0
        self.finished = False
        self.stepped = False
        self.task_dic = {
            0: {
                'ID': 0,
                'requirements': np.array([1, 0, 0, 0, 0]),
                'status': np.array([1, 0, 0, 0, 0]),
                'members': [],
                'finished': False,
                'feasible_assignment': False,
                'time_start': 0.0,
            }
        }
        self.agent_dic = {
            0: {
                'ID': 0,
                'species': 0,
                'abilities': np.array([1, 0, 0, 0, 0]),
                'current_task': -1,
                'current_action_index': 0,
                'no_choice': False,
                'sum_waiting_time': 0.0,
                'travel_dist': 0.0,
            }
        }

    def init_state(self):
        self.__init__()

    def next_decision(self):
        return ([0], []), self.current_time

    def agent_observe(self, agent_id, max_waiting=False):
        tasks = np.ones((1, 2, 15), dtype=float)
        agents = np.ones((1, 1, 11), dtype=float)
        mask = np.array([[False, False]])
        return tasks, agents, mask

    def agent_step(self, agent_id, action, decision_step):
        self.stepped = True
        self.agent_dic[agent_id]['current_task'] = action - 1
        self.task_dic[0]['status'] = np.zeros(5, dtype=int)
        self.task_dic[0]['finished'] = True
        return 0, True, []

    def check_finished(self):
        self.finished = self.stepped
        return self.finished

    def get_episode_reward(self, max_time):
        return -1.0, [self.task_dic[0]['finished']]

    @staticmethod
    def get_matrix(dictionary, key):
        return [value[key] for value in dictionary.values()]

    def get_efficiency(self):
        return 0.0

    def get_unfinished_tasks(self):
        return [
            not task['finished'] and np.any(task['status'] > 0)
            for task in self.task_dic.values()]

    def get_capability_match(self, agent_id, task_id):
        return 1.0

    def get_capability_match_action_vector(self, agent_id, action_count=None):
        values = np.zeros(action_count or self.tasks_num + 1, dtype=float)
        values[1] = 1.0
        return values


class DummyNetwork:
    def __init__(self):
        self.last_capability_match = None
        self.last_bias_params = None

    def __call__(self, tasks, agents, mask, index, return_details=False,
                 capability_match=None, bias_params=None):
        logits = torch.full(
            (tasks.shape[0], tasks.shape[1]),
            -100.0,
            dtype=tasks.dtype,
            device=tasks.device)
        logits[:, 1] = 0.0
        raw_logits = logits.clone()
        if capability_match is not None and bias_params is not None:
            self.last_capability_match = capability_match.detach().clone()
            self.last_bias_params = bias_params.detach().clone()
            logit_bias = compute_capability_match_bias(
                capability_match, ~mask.bool(), bias_params)
            logits = logits + logit_bias
        else:
            logit_bias = torch.zeros_like(raw_logits)
        biased_logits = logits.clone()
        logits[mask.bool()] = -1e8
        probs = torch.softmax(logits, dim=-1)
        logps = torch.log_softmax(logits, dim=-1)
        if return_details == 'all':
            return probs, logps, logits, raw_logits, logit_bias, biased_logits
        if return_details:
            return probs, logps, logits
        return probs, logps


class WorkerEvidenceTest(unittest.TestCase):
    def _run_once(self, enabled, bias_config=None):
        EvidenceParams.ENABLE_EVIDENCE_LOGGING = enabled
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
        network = DummyNetwork()
        with mock.patch('worker.TaskEnv', FakeEnv):
            worker = Worker(
                0,
                network,
                DummyNetwork(),
                global_step=4,
                device='cpu',
                env_params=[(1, 1), (1, 1), (1, 1)],
                bias_config=bias_config)
        reward, buffer, metrics = worker.run_episode(
            training=True,
            sample=True,
            max_waiting=False,
            episode_id=40)
        actions = [int(action.item()) for action in buffer[2]]
        return (
            reward,
            actions,
            metrics,
            copy.deepcopy(worker.last_episode_evidence),
            buffer,
            network)

    def test_logging_does_not_change_fixed_seed_transition(self):
        original = EvidenceParams.ENABLE_EVIDENCE_LOGGING
        try:
            reward_without, actions_without, metrics_without, evidence_without = (
                self._run_once(False)[:4])
            reward_with, actions_with, metrics_with, evidence_with = (
                self._run_once(True)[:4])
        finally:
            EvidenceParams.ENABLE_EVIDENCE_LOGGING = original

        self.assertEqual(reward_without, reward_with)
        self.assertEqual(actions_without, actions_with)
        self.assertEqual(metrics_without, metrics_with)
        self.assertIsNone(evidence_without)
        json.dumps(evidence_with, allow_nan=False)
        self.assertEqual(1, len(evidence_with['decisions']))
        self.assertLessEqual(
            len(evidence_with['decisions'][0]['candidates']),
            EvidenceParams.MAX_CANDIDATES_PER_DECISION)
        debug = evidence_with['decisions'][0]['decoder_logit_debug']
        self.assertIn('raw_decoder_logits', debug)
        self.assertIn('capability_logit_bias', debug)
        self.assertIn('biased_decoder_logits', debug)
        self.assertEqual(
            len(debug['raw_decoder_logits']),
            len(debug['biased_decoder_logits']))
        self.assertIn(
            'capability_logit_bias',
            evidence_with['decisions'][0]['candidates'][0])
        self.assertIn(
            'biased_model_logit',
            evidence_with['decisions'][0]['candidates'][0])
        self.assertTrue(evidence_with['episode']['success'])

    def test_disabled_bias_snapshot_preserves_fixed_seed_transition(self):
        original = EvidenceParams.ENABLE_EVIDENCE_LOGGING
        try:
            no_config = self._run_once(False)
            disabled_config = self._run_once(False, {
                'apply_bias': False,
                'used_weight': 1.5,
                'used_lambda': 0.9,
                'clip_range': [-0.2, 0.2],
            })
        finally:
            EvidenceParams.ENABLE_EVIDENCE_LOGGING = original

        self.assertEqual(no_config[:3], disabled_config[:3])
        self.assertIsNone(disabled_config[5].last_bias_params)

    def test_active_bias_snapshot_is_cached_in_every_transition(self):
        original = EvidenceParams.ENABLE_EVIDENCE_LOGGING
        snapshot = {
            'global_step': 3000,
            'source_report': 'evidence_window_00000000_00003000.md',
            'apply_bias': True,
            'used_weight': 0.42,
            'used_lambda': 0.06,
            'clip_range': [-0.5, 0.5],
        }
        try:
            _, actions, _, _, buffer, network = self._run_once(
                False, snapshot)
        finally:
            EvidenceParams.ENABLE_EVIDENCE_LOGGING = original

        self.assertEqual([1], actions)
        self.assertEqual(1, len(buffer[7]))
        self.assertEqual(1, len(buffer[8]))
        torch.testing.assert_close(
            buffer[7][0][:2], torch.tensor([0.0, 1.0]))
        expected_params = torch.tensor([1.0, 0.42, 0.06, -0.5, 0.5])
        torch.testing.assert_close(buffer[8][0], expected_params)
        torch.testing.assert_close(network.last_bias_params[0], expected_params)
        torch.testing.assert_close(
            network.last_capability_match[0, :2],
            torch.tensor([0.0, 1.0]))

    def test_timeout_and_deadlock_classification(self):
        with mock.patch('worker.TaskEnv', FakeEnv):
            worker = Worker(
                0,
                DummyNetwork(),
                DummyNetwork(),
                global_step=0,
                device='cpu',
                env_params=[(1, 1), (1, 1), (1, 1)])
        perf = {'waiting_time': [0.0]}

        worker.env.finished = False
        worker.env.current_time = worker.max_time
        timeout = worker._build_episode_evidence_summary(
            -1.0, [False], perf, [1], 5)
        self.assertTrue(timeout['timeout'])
        self.assertFalse(timeout['deadlock'])

        worker.env.current_time = worker.max_time - 1
        deadlock = worker._build_episode_evidence_summary(
            -1.0, [False], perf, [1], 5)
        self.assertTrue(deadlock['deadlock'])
        self.assertEqual(5, deadlock['deadlock_step'])


if __name__ == '__main__':
    unittest.main()
