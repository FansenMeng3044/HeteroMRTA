import json
import re
from pathlib import Path

from utils.deepseek_client import DeepSeekResponseTruncated


class DeepSeekBiasController:
    """Claim report windows once and update the main-process BiasManager."""

    WINDOW_PATTERN = re.compile(
        r'^evidence_window_(\d+)_(\d+)\.md$')

    def __init__(self, enabled, client, bias_manager, response_output_dir,
                 update_interval_steps=30000):
        self.enabled = bool(enabled)
        self.client = client
        self.bias_manager = bias_manager
        self.response_output_dir = Path(response_output_dir)
        self.update_interval_steps = int(update_interval_steps)
        if self.enabled:
            self.response_output_dir.mkdir(parents=True, exist_ok=True)
            restored_attempt_step = self._restore_last_attempt_end()
            restored_bias_step = int(
                self.bias_manager.get_snapshot().get('global_step') or 0)
            self.last_attempt_end_step = max(
                restored_attempt_step, restored_bias_step)
        else:
            self.last_attempt_end_step = 0

    def process_report(self, report_path):
        """Call DeepSeek at most once for each eligible update interval."""
        report_path = Path(report_path)
        window = self._parse_window(report_path)
        if not self.enabled or window is None:
            return self.bias_manager.get_snapshot(), False
        _, report_end = window
        update_start = self.last_attempt_end_step
        update_end = report_end
        if update_end - update_start < self.update_interval_steps:
            return self.bias_manager.get_snapshot(), False

        stem = '{:08d}_{:08d}'.format(update_start, update_end)
        attempt_path = self.response_output_dir / (
            'deepseek_attempt_{}.json'.format(stem))
        prompt_path = self.response_output_dir / (
            'deepseek_prompt_{}.txt'.format(stem))
        raw_path = self.response_output_dir / (
            'deepseek_raw_{}.json'.format(stem))
        sanitized_path = self.response_output_dir / (
            'deepseek_sanitized_{}.json'.format(stem))
        warning_path = self.response_output_dir / (
            'deepseek_warning_{}.txt'.format(stem))
        source_reports = self._collect_update_reports(
            report_path, update_start, update_end)
        source_report_names = [path.name for path in source_reports]

        if attempt_path.exists():
            self.last_attempt_end_step = max(
                self.last_attempt_end_step, update_end)
            return self.bias_manager.get_snapshot(), False

        try:
            self._claim_attempt(
                attempt_path, source_report_names, update_start, update_end)
        except FileExistsError:
            self.last_attempt_end_step = max(
                self.last_attempt_end_step, update_end)
            return self.bias_manager.get_snapshot(), False
        except Exception:
            return self.bias_manager.get_snapshot(), False

        self.last_attempt_end_step = update_end
        try:
            report_text = self._read_update_reports(
                source_reports, update_start, update_end)
            prompt = self.client.build_prompt(report_text)
            prompt_path.write_text(prompt, encoding='utf-8')
            _, raw_response, raw_config = self.client.request_bias_config(
                report_path, prompt=prompt)
            self._safe_write_json(raw_path, raw_response)
            sanitized, snapshot = self.bias_manager.update_from_config(
                raw_config,
                global_step=update_end,
                source_report='evidence_reports_{}.md'.format(stem))
            sanitized_output = dict(sanitized)
            sanitized_output.update({
                'status': 'applied',
                'source_reports': source_report_names,
                'active_bias': snapshot,
            })
            self._safe_write_json(sanitized_path, sanitized_output)
            self._safe_write_json(attempt_path, {
                'status': 'success',
                'source_report': (
                    source_report_names[-1] if source_report_names else None),
                'source_reports': source_report_names,
                'window_start': update_start,
                'window_end': update_end,
            })
            return snapshot, True
        except Exception as error:
            # Persist only the exception type to avoid leaking request headers.
            truncated = isinstance(error, DeepSeekResponseTruncated)
            warning = None
            if truncated:
                warning = (
                    'DeepSeek response was truncated at update window '
                    '{}-{}. Bias update was skipped and the previous safe '
                    'config remains active. Increase DEEPSEEK_MAX_TOKENS, '
                    'reduce report size, or keep thinking disabled.'
                ).format(update_start, update_end)
            failure = {
                'status': 'failed',
                'error_type': type(error).__name__,
                'finish_reason': getattr(error, 'finish_reason', None),
                'truncated': truncated,
                'warning': warning,
                'source_report': (
                    source_report_names[-1] if source_report_names else None),
                'source_reports': source_report_names,
                'window_start': update_start,
                'window_end': update_end,
            }
            self._safe_write_json(raw_path, failure)
            self._safe_write_json(sanitized_path, {
                'status': 'fallback',
                'truncated': truncated,
                'warning': warning,
                'source_reports': source_report_names,
                'active_bias': self.bias_manager.get_snapshot(),
            })
            self._safe_write_json(attempt_path, failure)
            if warning:
                self._safe_write_text(warning_path, warning + '\n')
                print('DeepSeek warning:', warning)
            return self.bias_manager.get_snapshot(), False

    @classmethod
    def _collect_update_reports(cls, report_path, update_start, update_end):
        reports = []
        for path in report_path.parent.glob('evidence_window_*.md'):
            window = cls._parse_window(path)
            if window is None:
                continue
            start_step, end_step = window
            if (start_step >= update_start and end_step <= update_end
                    and end_step > update_start):
                reports.append(path)
        if report_path not in reports:
            reports.append(report_path)
        reports.sort(key=lambda path: cls._parse_window(path) or (0, 0))
        return reports

    @staticmethod
    def _read_update_reports(report_paths, update_start, update_end):
        report_names = [path.name for path in report_paths]
        lines = [
            'DeepSeek bias update window: ({}, {}]'.format(
                update_start, update_end),
            '',
            'Included evidence reports:',
        ]
        lines.extend('- {}'.format(name) for name in report_names)
        for path in report_paths:
            lines.extend([
                '',
                '===== {} ====='.format(path.name),
                '',
                path.read_text(encoding='utf-8'),
            ])
        return '\n'.join(lines)

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
    def _claim_attempt(path, source_reports, start_step, end_step):
        value = {
            'status': 'started',
            'source_report': (
                source_reports[-1] if source_reports else None),
            'source_reports': source_reports,
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

    @staticmethod
    def _safe_write_text(path, value):
        try:
            path.write_text(value, encoding='utf-8')
        except Exception:
            pass
