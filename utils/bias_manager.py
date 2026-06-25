import json
import math
import re
from pathlib import Path

import torch


EXPLICIT_BIAS_FEATURES = (
    'completion_potential',
    'requirement_reduction_ratio',
    'travel_time',
    'waiting_pressure',
)


def zero_feature_weights():
    return {feature: 0.0 for feature in EXPLICIT_BIAS_FEATURES}


DISABLED_BIAS_SNAPSHOT = {
    'global_step': 0,
    'source_report': None,
    'apply_bias': False,
    'feature_names': list(EXPLICIT_BIAS_FEATURES),
    'used_weights': zero_feature_weights(),
    'used_lambda': 0.0,
    'clip_range': [-2.0, 2.0],
    'raw_deepseek_weights': None,
    'raw_deepseek_lambda': None,
    'ema_alpha': 0.3,
    'update_interval_steps': 30000,
}


def normalize_bias_snapshot(snapshot=None):
    """Return a detached JSON-safe bias snapshot with conservative defaults."""
    normalized = dict(DISABLED_BIAS_SNAPSHOT)
    normalized['used_weights'] = zero_feature_weights()
    if isinstance(snapshot, dict):
        normalized.update(snapshot)
    normalized['feature_names'] = list(EXPLICIT_BIAS_FEATURES)

    def finite_float(value, default):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return value if math.isfinite(value) else default

    requested_apply = bool(normalized.get('apply_bias'))
    used_weights = normalized.get('used_weights')
    if not isinstance(used_weights, dict):
        used_weights = {}
    normalized['used_weights'] = {
        feature: finite_float(used_weights.get(feature), 0.0)
        for feature in EXPLICIT_BIAS_FEATURES
    }
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
    raw_weights = normalized.get('raw_deepseek_weights')
    if isinstance(raw_weights, dict):
        normalized['raw_deepseek_weights'] = {
            feature: (
                finite_float(raw_weights.get(feature), None)
                if raw_weights.get(feature) is not None else None)
            for feature in EXPLICIT_BIAS_FEATURES
        }
    else:
        normalized['raw_deepseek_weights'] = None
    value = normalized.get('raw_deepseek_lambda')
    normalized['raw_deepseek_lambda'] = (
        finite_float(value, None) if value is not None else None)
    normalized['apply_bias'] = bool(
        requested_apply
        and any(value != 0.0 for value in normalized['used_weights'].values())
        and normalized['used_lambda'] != 0.0)
    return normalized


def bias_snapshot_to_tensor(snapshot, device='cpu', dtype=torch.float32):
    """Encode one immutable snapshot as [apply, lambda, low, high, weights...]."""
    snapshot = normalize_bias_snapshot(snapshot)
    values = [
        1.0 if snapshot['apply_bias'] else 0.0,
        snapshot['used_lambda'],
        snapshot['clip_range'][0],
        snapshot['clip_range'][1],
    ]
    values.extend(
        snapshot['used_weights'][feature]
        for feature in EXPLICIT_BIAS_FEATURES)
    return torch.tensor([values], dtype=dtype, device=device)


def compute_explicit_feature_bias(explicit_features, valid_mask, bias_params):
    """Compute clipped lambda * weighted explicit features for each action."""
    if explicit_features is None or bias_params is None:
        return None
    if bias_params.dim() == 1:
        bias_params = bias_params.unsqueeze(0)
    if explicit_features.dim() == 2:
        explicit_features = explicit_features.unsqueeze(-1)

    apply_bias = bias_params[:, 0].unsqueeze(1)
    feature_dim = explicit_features.size(-1)
    if bias_params.size(1) == 5 and feature_dim == 1:
        # Compatibility with the previous capability_match-only tensor:
        # [apply, weight, lambda, clip_low, clip_high].
        weights = bias_params[:, 1].view(-1, 1, 1)
        lambda_ = bias_params[:, 2].unsqueeze(1)
        clip_low = bias_params[:, 3].unsqueeze(1)
        clip_high = bias_params[:, 4].unsqueeze(1)
    else:
        weights = bias_params[:, 4:4 + feature_dim].unsqueeze(1)
        lambda_ = bias_params[:, 1].unsqueeze(1)
        clip_low = bias_params[:, 2].unsqueeze(1)
        clip_high = bias_params[:, 3].unsqueeze(1)
    feature_score = torch.sum(explicit_features * weights, dim=-1)
    raw_bias = apply_bias * lambda_ * feature_score
    clipped = torch.maximum(torch.minimum(raw_bias, clip_high), clip_low)
    return torch.where(valid_mask.bool(), clipped, torch.zeros_like(clipped))


def compute_capability_match_bias(capability_match, valid_mask, bias_params):
    """Backward-compatible wrapper for the old single-feature tests/callers."""
    if capability_match is None:
        return None
    return compute_explicit_feature_bias(
        capability_match.unsqueeze(-1), valid_mask, bias_params)


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
        unknown = set(weights) - set(EXPLICIT_BIAS_FEATURES)
        if unknown:
            raise ValueError(
                'unsupported explicit feature weights: {}'.format(
                    ', '.join(sorted(unknown))))

        raw_weights = {}
        sanitized_weights = {}
        for feature in EXPLICIT_BIAS_FEATURES:
            raw_weight = self._finite_number(
                weights.get(feature, 0.0), 'weights.{}'.format(feature))
            raw_weights[feature] = raw_weight
            sanitized_weights[feature] = self._clamp(
                raw_weight, self.weight_range)
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

        sanitized_lambda = self._clamp(raw_lambda, self.lambda_range)
        return {
            'weights': sanitized_weights,
            'lambda': sanitized_lambda,
            'clip_range': [clip_low, clip_high],
            'rationale': self._sanitize_rationale(
                raw_config.get('rationale')),
            'raw_deepseek_weights': raw_weights,
            'raw_deepseek_lambda': raw_lambda,
        }

    def update_from_config(self, raw_config, global_step, source_report):
        sanitized = self.sanitize(raw_config)
        previous_weights = self.current['used_weights']
        previous_lambda = self.current['used_lambda']
        raw_weights = sanitized['weights']
        raw_lambda = sanitized['lambda']
        used_weights = {
            feature: (
                self.ema_alpha * raw_weights[feature]
                + (1.0 - self.ema_alpha) * previous_weights[feature])
            for feature in EXPLICIT_BIAS_FEATURES
        }
        used_lambda = (
            self.ema_alpha * raw_lambda
            + (1.0 - self.ema_alpha) * previous_lambda)
        candidate = normalize_bias_snapshot({
            'global_step': int(global_step),
            'source_report': Path(source_report).name,
            'apply_bias': bool(
                self.enabled
                and any(value != 0.0 for value in used_weights.values())
                and used_lambda != 0.0),
            'feature_names': list(EXPLICIT_BIAS_FEATURES),
            'used_weights': used_weights,
            'used_lambda': used_lambda,
            'clip_range': sanitized['clip_range'],
            'raw_deepseek_weights': sanitized['raw_deepseek_weights'],
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
        snapshot['feature_names'] = list(EXPLICIT_BIAS_FEATURES)
        snapshot['used_weights'] = dict(self.current['used_weights'])
        if snapshot.get('raw_deepseek_weights') is not None:
            snapshot['raw_deepseek_weights'] = dict(
                snapshot['raw_deepseek_weights'])
        if not self.enabled:
            snapshot['apply_bias'] = False
            snapshot['used_weights'] = zero_feature_weights()
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
