import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from attention import AttentionNet
from parameters import DeepSeekBiasParams, EvidenceParams
from utils.bias_manager import (
    BiasManager,
    bias_snapshot_to_tensor,
    compute_capability_match_bias,
    normalize_bias_snapshot,
)
from utils.deepseek_bias_controller import DeepSeekBiasController
from utils.deepseek_client import DeepSeekClient, DeepSeekResponseTruncated


VALID_RAW_CONFIG = {
    'weights': {'capability_match': 0.8},
    'lambda': 0.12,
    'clip_range': [-2.0, 2.0],
    'rationale': {
        'main_failure_modes': ['low capability match'],
        'expected_effect': ['prefer contributable tasks'],
    },
}


class CapabilityBiasMathTest(unittest.TestCase):
    def test_formula_clipping_depot_and_invalid_actions(self):
        capability = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        valid = torch.tensor([[True, True, False, True]])
        params = torch.tensor([[1.0, 2.0, 0.5, -0.4, 0.4]])
        bias = compute_capability_match_bias(capability, valid, params)
        torch.testing.assert_close(
            bias, torch.tensor([[0.0, 0.4, 0.0, 0.0]]))

    def test_attention_no_bias_path_is_exact_and_mask_is_preserved(self):
        torch.manual_seed(5)
        network = AttentionNet(4, 5, 16)
        tasks = torch.rand(1, 4, 5) + 0.1
        agents = torch.rand(1, 3, 4) + 0.1
        mask = torch.tensor([[False, False, True, False]])
        index = torch.tensor([[[1]]])
        original_probs, original_logps = network(tasks, agents, mask, index)

        disabled = bias_snapshot_to_tensor({
            'apply_bias': False,
            'used_weight': 1.0,
            'used_lambda': 1.0,
        })
        capability = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        disabled_probs, disabled_logps = network(
            tasks,
            agents,
            mask,
            index,
            capability_match=capability,
            bias_params=disabled)
        self.assertTrue(torch.equal(original_probs, disabled_probs))
        self.assertTrue(torch.equal(original_logps, disabled_logps))

        active = bias_snapshot_to_tensor({
            'apply_bias': True,
            'used_weight': 1.0,
            'used_lambda': 1.0,
            'clip_range': [-2.0, 2.0],
        })
        active_probs, _ = network(
            tasks,
            agents,
            mask,
            index,
            capability_match=capability,
            bias_params=active)
        self.assertEqual(0.0, active_probs[0, 2].item())
        self.assertGreater(active_probs[0, 1].item(), original_probs[0, 1].item())


class BiasManagerTest(unittest.TestCase):
    def test_requested_defaults_are_exposed(self):
        self.assertTrue(EvidenceParams.ENABLE_EVIDENCE_LOGGING)
        self.assertEqual(3000, EvidenceParams.EVIDENCE_LOG_INTERVAL_STEPS)
        self.assertEqual('./evidence_logs', EvidenceParams.EVIDENCE_OUTPUT_DIR)
        self.assertEqual(20, EvidenceParams.MAX_CASES_PER_REPORT)
        self.assertEqual(8, EvidenceParams.MAX_CANDIDATES_PER_DECISION)
        self.assertTrue(DeepSeekBiasParams.ENABLE_DEEPSEEK_BIAS)
        self.assertEqual(
            30000, DeepSeekBiasParams.DEEPSEEK_BIAS_UPDATE_INTERVAL_STEPS)
        self.assertEqual(
            'https://api.deepseek.com',
            DeepSeekBiasParams.DEEPSEEK_BASE_URL)
        self.assertEqual(
            'deepseek-v4-pro', DeepSeekBiasParams.DEEPSEEK_MODEL)
        self.assertEqual(0.0, DeepSeekBiasParams.DEEPSEEK_TEMPERATURE)
        self.assertEqual(4096, DeepSeekBiasParams.DEEPSEEK_MAX_TOKENS)
        self.assertEqual(120, DeepSeekBiasParams.DEEPSEEK_TIMEOUT)
        self.assertTrue(DeepSeekBiasParams.DEEPSEEK_USE_JSON_RESPONSE)
        self.assertEqual(
            'disabled', DeepSeekBiasParams.DEEPSEEK_THINKING_TYPE)
        self.assertEqual(0.3, DeepSeekBiasParams.LLM_BIAS_EMA_ALPHA)

    def test_initial_off_ema_active_file_and_restore(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bias_dir = Path(temp_dir) / 'bias'
            response_dir = Path(temp_dir) / 'responses'
            manager = BiasManager(
                output_dir=bias_dir,
                response_output_dir=response_dir,
                ema_alpha=0.3)
            self.assertFalse(manager.get_snapshot()['apply_bias'])

            sanitized, snapshot = manager.update_from_config(
                VALID_RAW_CONFIG,
                global_step=3000,
                source_report='evidence_window_00000000_00003000.md')
            self.assertAlmostEqual(0.24, snapshot['used_weight'])
            self.assertAlmostEqual(0.036, snapshot['used_lambda'])
            self.assertTrue(snapshot['apply_bias'])
            self.assertEqual(0.8, snapshot['raw_deepseek_weight'])
            self.assertEqual(0.12, snapshot['raw_deepseek_lambda'])
            self.assertTrue(
                (bias_dir / 'active_bias_config_00003000.json').exists())

            restored = BiasManager(
                output_dir=bias_dir,
                response_output_dir=response_dir,
                ema_alpha=0.3)
            self.assertEqual(snapshot, restored.get_snapshot())
            self.assertEqual(
                {'capability_match': 0.8}, sanitized['weights'])

    def test_sanitize_rejects_other_features_and_clamps_safe_ranges(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BiasManager(output_dir=Path(temp_dir) / 'bias')
            with self.assertRaises(ValueError):
                manager.sanitize({
                    'weights': {
                        'capability_match': 1.0,
                        'sync_delay': 1.0,
                    },
                    'lambda': 1.0,
                    'clip_range': [-2.0, 2.0],
                })
            sanitized = manager.sanitize({
                'weights': {'capability_match': 99.0},
                'lambda': 99.0,
                'clip_range': [-99.0, 99.0],
            })
            self.assertEqual(2.0, sanitized['weights']['capability_match'])
            self.assertEqual(1.0, sanitized['lambda'])
            self.assertEqual([-10.0, 10.0], sanitized['clip_range'])

    def test_disabled_manager_creates_no_output_and_returns_zero_bias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / 'disabled'
            manager = BiasManager(enabled=False, output_dir=output)
            snapshot = manager.get_snapshot()
            self.assertFalse(snapshot['apply_bias'])
            self.assertEqual(0.0, snapshot['used_weight'])
            self.assertFalse(output.exists())

    def test_failed_active_config_write_keeps_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BiasManager(output_dir=Path(temp_dir) / 'bias')
            previous = manager.get_snapshot()
            with mock.patch.object(
                    manager, '_write_json', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    manager.update_from_config(
                        VALID_RAW_CONFIG,
                        3000,
                        'evidence_window_00000000_00003000.md')
            self.assertEqual(previous, manager.get_snapshot())

    def test_malformed_restored_snapshot_falls_back_to_zero_bias(self):
        malformed = normalize_bias_snapshot({
            'apply_bias': True,
            'used_weight': float('nan'),
            'used_lambda': 'bad',
            'clip_range': [5.0],
            'global_step': 'bad',
        })
        self.assertFalse(malformed['apply_bias'])
        self.assertEqual(0.0, malformed['used_weight'])
        self.assertEqual(0.0, malformed['used_lambda'])
        self.assertEqual([-2.0, 2.0], malformed['clip_range'])
        self.assertEqual(0, malformed['global_step'])


class DeepSeekClientTest(unittest.TestCase):
    def test_key_is_environment_only_and_json_response_is_parsed(self):
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update({
                'url': url,
                'headers': headers,
                'payload': payload,
                'timeout': timeout,
            })
            return {
                'choices': [{
                    'message': {'content': json.dumps(VALID_RAW_CONFIG)}
                }]
            }

        client = DeepSeekClient(
            'https://api.deepseek.com',
            'deepseek-v4-flash',
            thinking_type='disabled',
            transport=transport)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / 'report.md'
            report.write_text('evidence', encoding='utf-8')
            with mock.patch.dict(
                    os.environ, {'DEEPSEEK_API_KEY': 'secret-value'}):
                prompt, raw, config = client.request_bias_config(report)
        self.assertEqual(VALID_RAW_CONFIG, config)
        self.assertEqual(
            {'type': 'json_object'},
            captured['payload']['response_format'])
        self.assertEqual(
            {'type': 'disabled'},
            captured['payload']['thinking'])
        self.assertNotIn('secret-value', prompt)
        self.assertNotIn('secret-value', json.dumps(raw))
        self.assertEqual(
            'Bearer secret-value', captured['headers']['Authorization'])

    def test_missing_key_fails_before_transport(self):
        client = DeepSeekClient(
            'https://api.deepseek.com',
            'deepseek-v4-flash',
            transport=lambda *args: self.fail('transport must not run'))
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / 'report.md'
            report.write_text('evidence', encoding='utf-8')
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError):
                    client.request_bias_config(report)

    def test_length_finish_reason_is_rejected(self):
        def transport(url, headers, payload, timeout):
            return {
                'choices': [{
                    'finish_reason': 'length',
                    'message': {'content': json.dumps(VALID_RAW_CONFIG)},
                }]
            }

        client = DeepSeekClient(
            'https://api.deepseek.com',
            'deepseek-v4-flash',
            transport=transport)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / 'report.md'
            report.write_text('evidence', encoding='utf-8')
            with mock.patch.dict(
                    os.environ, {'DEEPSEEK_API_KEY': 'secret'}):
                with self.assertRaises(DeepSeekResponseTruncated):
                    client.request_bias_config(report)


class DeepSeekBiasControllerTest(unittest.TestCase):
    @staticmethod
    def _report(path, start=0, end=3000):
        report = path / (
            'evidence_window_{:08d}_{:08d}.md'.format(start, end))
        report.write_text('report evidence', encoding='utf-8')
        return report

    def test_reports_are_aggregated_until_update_interval(self):
        calls = []

        def transport(url, headers, payload, timeout):
            calls.append(payload)
            return {
                'choices': [{
                    'message': {'content': json.dumps(VALID_RAW_CONFIG)}
                }]
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            responses = root / 'responses'
            manager = BiasManager(
                output_dir=root / 'bias',
                response_output_dir=responses)
            client = DeepSeekClient(
                'https://api.deepseek.com',
                'deepseek-v4-flash',
                transport=transport)
            reports = [
                self._report(root, start, start + 3000)
                for start in range(0, 30000, 3000)
            ]
            with mock.patch.dict(
                    os.environ, {'DEEPSEEK_API_KEY': 'secret'}):
                controller = DeepSeekBiasController(
                    True, client, manager, responses, 30000)
                for report in reports[:-1]:
                    snapshot, updated = controller.process_report(report)
                    self.assertFalse(updated)
                    self.assertFalse(snapshot['apply_bias'])
                self.assertEqual([], calls)

                snapshot, updated = controller.process_report(reports[-1])

            self.assertTrue(updated)
            self.assertTrue(snapshot['apply_bias'])
            self.assertEqual(1, len(calls))
            prompt = calls[0]['messages'][0]['content']
            self.assertIn('DeepSeek bias update window: (0, 30000]', prompt)
            self.assertIn(
                'evidence_window_00000000_00003000.md', prompt)
            self.assertIn(
                'evidence_window_00027000_00030000.md', prompt)
            attempt = json.loads((
                responses / 'deepseek_attempt_00000000_00030000.json'
            ).read_text(encoding='utf-8'))
            sanitized = json.loads((
                responses / 'deepseek_sanitized_00000000_00030000.json'
            ).read_text(encoding='utf-8'))
            expected_sources = [report.name for report in reports]
            self.assertEqual(expected_sources, attempt['source_reports'])
            self.assertEqual(expected_sources, sanitized['source_reports'])
            self.assertTrue((
                root / 'bias' / 'active_bias_config_00030000.json'
            ).exists())

    def test_one_call_per_window_and_restart_dedup(self):
        calls = []

        def transport(url, headers, payload, timeout):
            calls.append(payload)
            return {
                'choices': [{
                    'message': {'content': json.dumps(VALID_RAW_CONFIG)}
                }]
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            responses = root / 'responses'
            manager = BiasManager(
                output_dir=root / 'bias',
                response_output_dir=responses)
            client = DeepSeekClient(
                'https://api.deepseek.com',
                'deepseek-v4-flash',
                transport=transport)
            report = self._report(root)
            with mock.patch.dict(
                    os.environ, {'DEEPSEEK_API_KEY': 'secret'}):
                controller = DeepSeekBiasController(
                    True, client, manager, responses, 3000)
                snapshot, updated = controller.process_report(report)
                self.assertTrue(updated)
                controller.process_report(report)
                restarted = DeepSeekBiasController(
                    True, client, manager, responses, 3000)
                restarted.process_report(report)

            self.assertEqual(1, len(calls))
            self.assertTrue(snapshot['apply_bias'])
            expected = [
                'deepseek_attempt_00000000_00003000.json',
                'deepseek_prompt_00000000_00003000.txt',
                'deepseek_raw_00000000_00003000.json',
                'deepseek_sanitized_00000000_00003000.json',
            ]
            self.assertEqual(
                expected, sorted(path.name for path in responses.iterdir()))
            all_text = ''.join(
                path.read_text(encoding='utf-8')
                for path in responses.iterdir())
            self.assertNotIn('secret', all_text)

    def test_failure_keeps_previous_safe_config_and_is_not_retried(self):
        calls = []

        def failing_transport(*args):
            calls.append(1)
            raise TimeoutError('do not persist this message')

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            responses = root / 'responses'
            manager = BiasManager(
                output_dir=root / 'bias',
                response_output_dir=responses)
            manager.update_from_config(
                VALID_RAW_CONFIG, 3000,
                'evidence_window_00000000_00003000.md')
            previous = manager.get_snapshot()
            report = self._report(root, 3000, 6000)
            client = DeepSeekClient(
                'https://api.deepseek.com',
                'deepseek-v4-flash',
                transport=failing_transport)
            with mock.patch.dict(
                    os.environ, {'DEEPSEEK_API_KEY': 'secret'}):
                controller = DeepSeekBiasController(
                    True, client, manager, responses, 3000)
                snapshot, updated = controller.process_report(report)
                controller.process_report(report)

            self.assertFalse(updated)
            self.assertEqual(previous, snapshot)
            self.assertEqual(1, len(calls))
            raw = json.loads((
                responses / 'deepseek_raw_00003000_00006000.json'
            ).read_text(encoding='utf-8'))
            self.assertEqual('TimeoutError', raw['error_type'])
            self.assertNotIn('do not persist', json.dumps(raw))

    def test_invalid_json_first_window_stays_disabled(self):
        calls = []

        def transport(*args):
            calls.append(1)
            return {'choices': [{'message': {'content': 'not json'}}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            responses = root / 'responses'
            manager = BiasManager(
                output_dir=root / 'bias',
                response_output_dir=responses)
            client = DeepSeekClient(
                'https://api.deepseek.com',
                'deepseek-v4-flash',
                transport=transport)
            controller = DeepSeekBiasController(
                True, client, manager, responses, 3000)
            report = self._report(root)
            with mock.patch.dict(
                    os.environ, {'DEEPSEEK_API_KEY': 'secret'}):
                snapshot, updated = controller.process_report(report)

            self.assertFalse(updated)
            self.assertFalse(snapshot['apply_bias'])
            self.assertEqual(0.0, snapshot['used_weight'])
            self.assertEqual(1, len(calls))

    def test_truncated_response_writes_visible_warning_and_fallback(self):
        def transport(*args):
            return {
                'choices': [{
                    'finish_reason': 'length',
                    'message': {'content': '{"weights":'},
                }]
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            responses = root / 'responses'
            manager = BiasManager(
                output_dir=root / 'bias',
                response_output_dir=responses)
            client = DeepSeekClient(
                'https://api.deepseek.com',
                'deepseek-v4-flash',
                transport=transport)
            controller = DeepSeekBiasController(
                True, client, manager, responses, 3000)
            report = self._report(root)
            with mock.patch.dict(
                    os.environ, {'DEEPSEEK_API_KEY': 'secret'}):
                snapshot, updated = controller.process_report(report)

            self.assertFalse(updated)
            self.assertFalse(snapshot['apply_bias'])
            raw = json.loads((
                responses / 'deepseek_raw_00000000_00003000.json'
            ).read_text(encoding='utf-8'))
            sanitized = json.loads((
                responses / 'deepseek_sanitized_00000000_00003000.json'
            ).read_text(encoding='utf-8'))
            attempt = json.loads((
                responses / 'deepseek_attempt_00000000_00003000.json'
            ).read_text(encoding='utf-8'))
            warning = (
                responses / 'deepseek_warning_00000000_00003000.txt')
            self.assertTrue(raw['truncated'])
            self.assertTrue(sanitized['truncated'])
            self.assertTrue(attempt['truncated'])
            self.assertEqual('length', raw['finish_reason'])
            self.assertTrue(warning.exists())
            self.assertIn('truncated', warning.read_text(encoding='utf-8'))

    def test_missing_report_is_a_safe_claimed_failure(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            responses = root / 'responses'
            manager = BiasManager(
                output_dir=root / 'bias',
                response_output_dir=responses)
            client = DeepSeekClient(
                'https://api.deepseek.com',
                'deepseek-v4-flash',
                transport=lambda *args: calls.append(1))
            controller = DeepSeekBiasController(
                True, client, manager, responses, 3000)
            missing = (
                root / 'evidence_window_00000000_00003000.md')
            snapshot, updated = controller.process_report(missing)

            self.assertFalse(updated)
            self.assertFalse(snapshot['apply_bias'])
            self.assertEqual([], calls)
            self.assertTrue((
                responses
                / 'deepseek_attempt_00000000_00003000.json'
            ).exists())


class ForwardNetworkIsolationTest(unittest.TestCase):
    def test_api_symbols_do_not_appear_in_model_or_worker_modules(self):
        root = Path(__file__).resolve().parents[1]
        for relative in ('attention.py', 'worker.py', 'runner.py'):
            source = (root / relative).read_text(encoding='utf-8')
            self.assertNotIn('DeepSeekClient', source)
            self.assertNotIn('request_bias_config', source)
            self.assertNotIn('urllib.request', source)


if __name__ == '__main__':
    unittest.main()
