import json
import math
import re
from pathlib import Path

import torch


DISABLED_BIAS_SNAPSHOT = {
    'global_step': 0,
    'source_report': None,
    'apply_bias': False,
    'used_weight': 0.0,
    'used_lambda': 0.0,
    'clip_range': [-2.0, 2.0],
    'raw_deepseek_weight': None,
    'raw_deepseek_lambda': None,
    'ema_alpha': 0.3,
    'update_interval_steps': 30000,
}


def normalize_bias_snapshot(snapshot=None):
    """Return a detached JSON-safe bias snapshot with conservative defaults."""
    normalized = dict(DISABLED_BIAS_SNAPSHOT)
    if isinstance(snapshot, dict):
        normalized.update(snapshot)

    def finite_float(value, default):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return value if math.isfinite(value) else default

    requested_apply = bool(normalized.get('apply_bias'))
    normalized['used_weight'] = finite_float(
        normalized.get('used_weight'), 0.0)
    normalized['used_lambda'] = finite_float(
        normalized.get('used_lambda'), 0.0)
    clip_range = normalized.get('clip_range') or [-2.0, 2.0]
    if not isinstance(clip_range, (list, tuple)) or len(clip_range) != 2:
        clip_range = [-2.0, 2.0]
    clip_low = finite_float(clip_range[0], -2.0)
    clip_high = finite_float(clip_range[1], 2.0)
    if clip_low >= clip_high:
        clip_low, clip_high = -2.0, 2.0
        requested_apply = False
    normalized['clip_range'] = [clip_low, clip_high]
    try:
        normalized['global_step'] = max(
            0, int(normalized.get('global_step') or 0))
    except (TypeError, ValueError, OverflowError):
        normalized['global_step'] = 0
        requested_apply = False
    try:
        normalized['update_interval_steps'] = max(
            1, int(normalized.get('update_interval_steps') or 30000))
    except (TypeError, ValueError, OverflowError):
        normalized['update_interval_steps'] = 30000
    normalized['ema_alpha'] = finite_float(
        normalized.get('ema_alpha'), 0.3)
    for field in ('raw_deepseek_weight', 'raw_deepseek_lambda'):
        value = normalized.get(field)
        normalized[field] = (
            finite_float(value, None) if value is not None else None)
    normalized['apply_bias'] = bool(
        requested_apply
        and normalized['used_weight'] != 0.0
        and normalized['used_lambda'] != 0.0)
    return normalized


def bias_snapshot_to_tensor(snapshot, device='cpu', dtype=torch.float32):
    """Encode one immutable snapshot as [apply, weight, lambda, low, high]."""
    snapshot = normalize_bias_snapshot(snapshot)
    return torch.tensor([[
        1.0 if snapshot['apply_bias'] else 0.0,
        snapshot['used_weight'],
        snapshot['used_lambda'],
        snapshot['clip_range'][0],
        snapshot['clip_range'][1],
    ]], dtype=dtype, device=device)


def compute_capability_match_bias(capability_match, valid_mask, bias_params):
    """Compute clip(lambda * weight * capability_match) for each action."""
    if capability_match is None or bias_params is None:
        return None
    if bias_params.dim() == 1:
        bias_params = bias_params.unsqueeze(0)
    if capability_match.dim() == 1:
        capability_match = capability_match.unsqueeze(0)

    apply_bias = bias_params[:, 0].unsqueeze(1)
    weight = bias_params[:, 1].unsqueeze(1)
    lambda_ = bias_params[:, 2].unsqueeze(1)
    clip_low = bias_params[:, 3].unsqueeze(1)
    clip_high = bias_params[:, 4].unsqueeze(1)
    raw_bias = apply_bias * lambda_ * weight * capability_match
    clipped = torch.maximum(torch.minimum(raw_bias, clip_high), clip_low)
    return torch.where(valid_mask.bool(), clipped, torch.zeros_like(clipped))


class BiasManager:
    """Sanitize, smooth, persist, and restore the active bias snapshot."""

    def __init__(self, enabled=True, output_dir='./evidence_logs/bias_configs',
                 response_output_dir='./evidence_logs/deepseek_responses',
                 ema_alpha=0.3, update_interval_steps=30000,
                 weight_range=(-2.0, 2.0), lambda_range=(0.0, 1.0),
                 clip_bound_range=(-10.0, 10.0)):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.response_output_dir = Path(response_output_dir)
        self.ema_alpha = float(ema_alpha)
        self.update_interval_steps = int(update_interval_steps)
        self.weight_range = tuple(weight_range)
        self.lambda_range = tuple(lambda_range)
        self.clip_bound_range = tuple(clip_bound_range)
        self.current = normalize_bias_snapshot({
            'ema_alpha': self.ema_alpha,
            'update_interval_steps': self.update_interval_steps,
        })
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.response_output_dir.mkdir(parents=True, exist_ok=True)
            self._restore_latest()

    @staticmethod
    def _finite_number(value, field):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('{} must be a number'.format(field))
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('{} must be finite'.format(field))
        return value

    @staticmethod
    def _clamp(value, bounds):
        return min(max(value, bounds[0]), bounds[1])

    @staticmethod
    def _sanitize_rationale(value):
        if not isinstance(value, dict):
            value = {}
        sanitized = {}
        for field in ('main_failure_modes', 'expected_effect'):
            items = value.get(field, [])
            if not isinstance(items, list):
                items = []
            sanitized[field] = [
                item[:500] for item in items[:20] if isinstance(item, str)
            ]
        return sanitized

    def sanitize(self, raw_config):
        if not isinstance(raw_config, dict):
            raise ValueError('DeepSeek output must be a JSON object')
        weights = raw_config.get('weights')
        if not isinstance(weights, dict):
            raise ValueError('weights must be an object')
        if set(weights) != {'capability_match'}:
            raise ValueError('only capability_match weight is allowed')

        raw_weight = self._finite_number(
            weights['capability_match'], 'weights.capability_match')
        raw_lambda = self._finite_number(raw_config.get('lambda'), 'lambda')
        clip_range = raw_config.get('clip_range')
        if not isinstance(clip_range, list) or len(clip_range) != 2:
            raise ValueError('clip_range must contain two values')
        clip_low = self._finite_number(clip_range[0], 'clip_range[0]')
        clip_high = self._finite_number(clip_range[1], 'clip_range[1]')
        clip_low = self._clamp(clip_low, self.clip_bound_range)
        clip_high = self._clamp(clip_high, self.clip_bound_range)
        if clip_low >= clip_high:
            raise ValueError('clip_range must satisfy low < high')

        sanitized_weight = self._clamp(raw_weight, self.weight_range)
        sanitized_lambda = self._clamp(raw_lambda, self.lambda_range)
        return {
            'weights': {'capability_match': sanitized_weight},
            'lambda': sanitized_lambda,
            'clip_range': [clip_low, clip_high],
            'rationale': self._sanitize_rationale(
                raw_config.get('rationale')),
            'raw_deepseek_weight': raw_weight,
            'raw_deepseek_lambda': raw_lambda,
        }

    def update_from_config(self, raw_config, global_step, source_report):
        sanitized = self.sanitize(raw_config)
        previous_weight = self.current['used_weight']
        previous_lambda = self.current['used_lambda']
        raw_weight = sanitized['weights']['capability_match']
        raw_lambda = sanitized['lambda']
        used_weight = (
            self.ema_alpha * raw_weight
            + (1.0 - self.ema_alpha) * previous_weight)
        used_lambda = (
            self.ema_alpha * raw_lambda
            + (1.0 - self.ema_alpha) * previous_lambda)
        candidate = normalize_bias_snapshot({
            'global_step': int(global_step),
            'source_report': Path(source_report).name,
            'apply_bias': bool(
                self.enabled and used_weight != 0.0 and used_lambda != 0.0),
            'used_weight': used_weight,
            'used_lambda': used_lambda,
            'clip_range': sanitized['clip_range'],
            'raw_deepseek_weight': sanitized['raw_deepseek_weight'],
            'raw_deepseek_lambda': sanitized['raw_deepseek_lambda'],
            'ema_alpha': self.ema_alpha,
            'update_interval_steps': self.update_interval_steps,
        })
        active_path = self.output_dir / (
            'active_bias_config_{:08d}.json'.format(int(global_step)))
        self._write_json(active_path, candidate)
        self.current = candidate
        return sanitized, self.get_snapshot()

    def get_snapshot(self):
        snapshot = dict(self.current)
        snapshot['clip_range'] = list(self.current['clip_range'])
        if not self.enabled:
            snapshot['apply_bias'] = False
            snapshot['used_weight'] = 0.0
            snapshot['used_lambda'] = 0.0
        return snapshot

    def _restore_latest(self):
        pattern = re.compile(r'^active_bias_config_(\d+)\.json$')
        latest = None
        for path in self.output_dir.glob('active_bias_config_*.json'):
            match = pattern.match(path.name)
            if match and (latest is None or int(match.group(1)) > latest[0]):
                latest = (int(match.group(1)), path)
        if latest is None:
            return
        try:
            restored = json.loads(latest[1].read_text(encoding='utf-8'))
            self.current = normalize_bias_snapshot(restored)
        except (OSError, ValueError, TypeError, KeyError):
            self.current = normalize_bias_snapshot({
                'ema_alpha': self.ema_alpha,
                'update_interval_steps': self.update_interval_steps,
            })

    @staticmethod
    def _write_json(path, value):
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            + '\n',
            encoding='utf-8')
