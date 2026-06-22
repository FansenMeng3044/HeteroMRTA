import json
import re
from pathlib import Path


class DeepSeekBiasController:
    """Claim report windows once and update the main-process BiasManager."""

    WINDOW_PATTERN = re.compile(
        r'^evidence_window_(\d+)_(\d+)\.md$')

    def __init__(self, enabled, client, bias_manager, response_output_dir,
                 update_interval_steps=3000):
        self.enabled = bool(enabled)
        self.client = client
        self.bias_manager = bias_manager
        self.response_output_dir = Path(response_output_dir)
        self.update_interval_steps = int(update_interval_steps)
        if self.enabled:
            self.response_output_dir.mkdir(parents=True, exist_ok=True)
            self.last_attempt_end_step = self._restore_last_attempt_end()
        else:
            self.last_attempt_end_step = 0

    def process_report(self, report_path):
        """Call DeepSeek at most once for an eligible newly generated report."""
        report_path = Path(report_path)
        window = self._parse_window(report_path)
        if not self.enabled or window is None:
            return self.bias_manager.get_snapshot(), False
        start_step, end_step = window
        if end_step - self.last_attempt_end_step < self.update_interval_steps:
            return self.bias_manager.get_snapshot(), False

        stem = '{:08d}_{:08d}'.format(start_step, end_step)
        attempt_path = self.response_output_dir / (
            'deepseek_attempt_{}.json'.format(stem))
        prompt_path = self.response_output_dir / (
            'deepseek_prompt_{}.txt'.format(stem))
        raw_path = self.response_output_dir / (
            'deepseek_raw_{}.json'.format(stem))
        sanitized_path = self.response_output_dir / (
            'deepseek_sanitized_{}.json'.format(stem))

        if attempt_path.exists():
            self.last_attempt_end_step = max(
                self.last_attempt_end_step, end_step)
            return self.bias_manager.get_snapshot(), False

        try:
            self._claim_attempt(
                attempt_path, report_path.name, start_step, end_step)
        except FileExistsError:
            self.last_attempt_end_step = max(
                self.last_attempt_end_step, end_step)
            return self.bias_manager.get_snapshot(), False
        except Exception:
            return self.bias_manager.get_snapshot(), False

        self.last_attempt_end_step = end_step
        try:
            report_text = report_path.read_text(encoding='utf-8')
            prompt = self.client.build_prompt(report_text)
            prompt_path.write_text(prompt, encoding='utf-8')
            _, raw_response, raw_config = self.client.request_bias_config(
                report_path, prompt=prompt)
            self._safe_write_json(raw_path, raw_response)
            sanitized, snapshot = self.bias_manager.update_from_config(
                raw_config,
                global_step=end_step,
                source_report=report_path)
            sanitized_output = dict(sanitized)
            sanitized_output.update({
                'status': 'applied',
                'active_bias': snapshot,
            })
            self._safe_write_json(sanitized_path, sanitized_output)
            self._safe_write_json(attempt_path, {
                'status': 'success',
                'source_report': report_path.name,
                'window_start': start_step,
                'window_end': end_step,
            })
            return snapshot, True
        except Exception as error:
            # Persist only the exception type to avoid leaking request headers.
            failure = {
                'status': 'failed',
                'error_type': type(error).__name__,
                'source_report': report_path.name,
                'window_start': start_step,
                'window_end': end_step,
            }
            self._safe_write_json(raw_path, failure)
            self._safe_write_json(sanitized_path, {
                'status': 'fallback',
                'active_bias': self.bias_manager.get_snapshot(),
            })
            self._safe_write_json(attempt_path, failure)
            return self.bias_manager.get_snapshot(), False

    def _restore_last_attempt_end(self):
        latest = 0
        pattern = re.compile(r'^deepseek_attempt_(\d+)_(\d+)\.json$')
        for path in self.response_output_dir.glob('deepseek_attempt_*.json'):
            match = pattern.match(path.name)
            if match:
                latest = max(latest, int(match.group(2)))
        return latest

    @classmethod
    def _parse_window(cls, report_path):
        match = cls.WINDOW_PATTERN.match(report_path.name)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _claim_attempt(path, source_report, start_step, end_step):
        value = {
            'status': 'started',
            'source_report': source_report,
            'window_start': start_step,
            'window_end': end_step,
        }
        with path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(
                value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    @staticmethod
    def _write_json(path, value):
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            + '\n',
            encoding='utf-8')

    @classmethod
    def _safe_write_json(cls, path, value):
        try:
            cls._write_json(path, value)
        except Exception:
            pass
