import copy
import json
import math
import re
from pathlib import Path
from utils.bias_manager import EXPLICIT_BIAS_FEATURES, zero_feature_weights


def to_python(value):
    """Convert tensors/NumPy values and containers to JSON-safe Python data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'tolist'):
        value = value.tolist()
    elif hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def get_task_state(task):
    """Describe task state without changing environment state."""
    if task.get('finished'):
        return 'finished'
    if task.get('feasible_assignment'):
        return 'assigned'
    if task.get('members'):
        return 'waiting'
    return 'open'


def select_candidate_action_indices(probabilities, valid_actions, capability_matches,
                                    chosen_action, max_candidates, top_k=5,
                                    low_threshold=0.3):
    """Select a compact, deterministic union of informative candidate actions."""
    action_count = min(len(probabilities), len(valid_actions), len(capability_matches))
    valid_indices = [index for index in range(action_count) if valid_actions[index]]
    ranked_by_probability = sorted(
        valid_indices, key=lambda index: (-probabilities[index], index))
    task_indices = [index for index in valid_indices
                    if capability_matches[index] is not None]
    ranked_by_match = sorted(
        task_indices,
        key=lambda index: (-capability_matches[index], -probabilities[index], index))

    selected = []

    def add(index):
        if index is not None and 0 <= index < action_count and index not in selected:
            selected.append(index)

    add(chosen_action)
    for index in ranked_by_probability[:top_k]:
        add(index)

    best_match = ranked_by_match[0] if ranked_by_match else None
    chosen_match = (capability_matches[chosen_action]
                    if 0 <= chosen_action < action_count else None)
    if chosen_match is None or chosen_match < low_threshold:
        add(best_match)
    else:
        add(best_match)

    for index in ranked_by_match:
        if capability_matches[index] > 0:
            add(index)

    return selected[:max_candidates]


def build_candidate_records(env, agent_id, chosen_action, probabilities, logits, mask,
                            max_candidates, top_k=5, low_threshold=0.3,
                            logit_bias=None, biased_logits=None,
                            explicit_features=None,
                            feature_names=EXPLICIT_BIAS_FEATURES):
    """Build bounded candidate records from the current decoder output and state."""
    action_count = env.tasks_num + 1
    probability_values = to_python(probabilities)[:action_count]
    logit_values = to_python(logits)[:action_count]
    logit_bias_values = (
        to_python(logit_bias)[:action_count]
        if logit_bias is not None else [None] * action_count)
    biased_logit_values = (
        to_python(biased_logits)[:action_count]
        if biased_logits is not None else [None] * action_count)
    mask_values = [bool(value) for value in to_python(mask)[:action_count]]
    valid_actions = [not value for value in mask_values]
    capability_matches = [None]
    capability_matches.extend(
        env.get_capability_match(agent_id, task_id)
        for task_id in range(env.tasks_num))
    feature_names = list(feature_names)
    feature_values = (
        to_python(explicit_features)[:action_count]
        if explicit_features is not None else None)

    def action_feature_dict(action_index):
        if feature_values is None or action_index >= len(feature_values):
            return {}
        row = feature_values[action_index]
        if row is None:
            return {}
        return {
            feature: row[index]
            for index, feature in enumerate(feature_names)
            if index < len(row)
        }

    selected = select_candidate_action_indices(
        probability_values,
        valid_actions,
        capability_matches,
        chosen_action,
        max_candidates,
        top_k=top_k,
        low_threshold=low_threshold)

    candidates = []
    for action_index in selected:
        if action_index == 0:
            candidates.append({
                'action_index': 0,
                'task_id': None,
                'task_state': 'depot',
                'is_valid': valid_actions[action_index],
                'is_chosen': action_index == chosen_action,
                'model_logit': logit_values[action_index],
                'explicit_feature_logit_bias': logit_bias_values[action_index],
                'capability_logit_bias': logit_bias_values[action_index],
                'biased_model_logit': biased_logit_values[action_index],
                'model_prob': probability_values[action_index],
                'remaining_requirement': None,
                'original_requirement': None,
                'explicit_features': action_feature_dict(action_index),
                # Depot is not a robot-task pair, so capability_match is null.
                'capability_match': None,
            })
            continue

        task = env.task_dic[action_index - 1]
        candidates.append({
            'action_index': action_index,
            'task_id': task['ID'],
            'task_state': get_task_state(task),
            'is_valid': valid_actions[action_index],
            'is_chosen': action_index == chosen_action,
            'model_logit': logit_values[action_index],
            'explicit_feature_logit_bias': logit_bias_values[action_index],
            'capability_logit_bias': logit_bias_values[action_index],
            'biased_model_logit': biased_logit_values[action_index],
            'model_prob': probability_values[action_index],
            'remaining_requirement': to_python(task['status']),
            'original_requirement': to_python(task['requirements']),
            'explicit_features': action_feature_dict(action_index),
            'capability_match': capability_matches[action_index],
        })
    return candidates


class EpisodeEvidenceBuffer:
    """Worker-local in-memory buffer; it never writes files."""

    def __init__(self, episode_id):
        self.episode_id = int(episode_id)
        self.decisions = []

    def record_decision(self, record):
        record = to_python(record)
        record['episode_id'] = self.episode_id
        record['decision_id'] = len(self.decisions)
        self.decisions.append(record)

    def record_episode_end(self, summary):
        summary = to_python(summary)
        summary['episode_id'] = self.episode_id
        outcome = {
            'eventual_episode_success': summary.get('success'),
            'eventual_deadlock': summary.get('deadlock'),
            'eventual_makespan': summary.get('final_makespan'),
            'eventual_awt': summary.get('average_waiting_time'),
            'eventual_awar': summary.get('average_wasted_ability_ratio'),
        }
        for decision in self.decisions:
            decision.update(outcome)
        summary['num_decisions'] = len(self.decisions)
        return {
            'episode': summary,
            'decisions': self.decisions,
        }


class EvidenceRecorder:
    """Main-process writer and window aggregator."""

    def __init__(self, enabled=True, interval_steps=10000,
                 output_dir='./evidence_logs', max_cases_per_report=5,
                 low_threshold=0.3, better_alternative_gap=0.3,
                 deadlock_lookback=10):
        self.enabled = bool(enabled)
        self.interval_steps = max(1, int(interval_steps))
        self.output_dir = Path(output_dir)
        self.max_cases_per_report = max(1, int(max_cases_per_report))
        self.low_threshold = float(low_threshold)
        self.better_alternative_gap = float(better_alternative_gap)
        self.deadlock_lookback = max(1, int(deadlock_lookback))
        self.global_step = 0
        self.window_start = 0
        self.decisions = []
        self.episodes = []
        self.previous_global_trend = None
        self.just_generated_report = False
        self.latest_markdown_report_path = None
        self.generated_markdown_report_paths = []
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._restore_existing_window_state()

    def _restore_existing_window_state(self):
        """Continue after the latest completed report instead of overwriting it."""
        pattern = re.compile(
            r'^evidence_window_(\d+)_(\d+)\.json$')
        latest = None
        for path in self.output_dir.glob('evidence_window_*.json'):
            match = pattern.match(path.name)
            if not match:
                continue
            start_step = int(match.group(1))
            end_step = int(match.group(2))
            if latest is None or end_step > latest[0]:
                latest = (end_step, start_step, path)
        if latest is None:
            return
        self.global_step = latest[0]
        is_partial = latest[0] - latest[1] < self.interval_steps
        self.window_start = latest[1] if is_partial else latest[0]
        try:
            report = json.loads(latest[2].read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            # Existing evidence files must never prevent training from starting.
            report = {}
        if is_partial:
            stem = '{:08d}_{:08d}'.format(latest[1], latest[0])
            self.decisions = self._read_jsonl(
                self.output_dir / 'decisions_{}.jsonl'.format(stem))
            self.episodes = self._read_jsonl(
                self.output_dir / 'episodes_{}.jsonl'.format(stem))
            self.previous_global_trend = (
                self._previous_complete_trend(latest[1]))
        else:
            self.previous_global_trend = report.get('global_training_trend')

    def _previous_complete_trend(self, before_step):
        pattern = re.compile(r'^evidence_window_(\d+)_(\d+)\.json$')
        latest = None
        for path in self.output_dir.glob('evidence_window_*.json'):
            match = pattern.match(path.name)
            if not match:
                continue
            start_step, end_step = map(int, match.groups())
            if (end_step - start_step == self.interval_steps
                    and end_step <= before_step
                    and (latest is None or end_step > latest[0])):
                latest = (end_step, path)
        if latest is None:
            return None
        try:
            report = json.loads(latest[1].read_text(encoding='utf-8'))
            return report.get('global_training_trend')
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _read_jsonl(path):
        records = []
        try:
            with path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
        except (OSError, ValueError, TypeError):
            return []
        return records

    def record_decision(self, record):
        if not self.enabled:
            return None
        self.global_step += 1
        record = to_python(copy.deepcopy(record))
        record['train_step'] = self.global_step
        self.decisions.append(record)
        return self.global_step

    def record_episode_end(self, record):
        if not self.enabled:
            return
        self.episodes.append(to_python(copy.deepcopy(record)))

    def record_episode_payload(self, payload):
        """Assign main-process steps to one completed worker episode."""
        if not self.enabled or not payload:
            return
        episode = copy.deepcopy(payload.get('episode', {}))
        episode_id = episode.get('episode_id')
        start_step = self.global_step + 1
        for decision in payload.get('decisions', []):
            decision = copy.deepcopy(decision)
            if episode_id is not None and decision.get('episode_id') is None:
                decision['episode_id'] = episode_id
            self.record_decision(decision)
        end_step = self.global_step
        episode['train_step_start'] = start_step if end_step >= start_step else end_step
        episode['train_step_end'] = end_step
        self.record_episode_end(episode)
        return self.flush_window_if_needed(self.global_step)

    def flush_window_if_needed(self, global_step=None, force=False):
        if not self.enabled:
            return []
        self.just_generated_report = False
        self.generated_markdown_report_paths = []
        if global_step is None:
            global_step = self.global_step
        written = []
        while global_step >= self.window_start + self.interval_steps:
            window_end = self.window_start + self.interval_steps
            written.extend(self._write_window(self.window_start, window_end))
            self.window_start = window_end
        if force and (global_step > self.window_start or self.decisions or self.episodes):
            written.extend(self._write_window(self.window_start, global_step))
            self.window_start = global_step
        markdown_paths = [
            path for path in written if str(path).lower().endswith('.md')]
        if markdown_paths:
            self.just_generated_report = True
            self.latest_markdown_report_path = markdown_paths[-1]
            self.generated_markdown_report_paths = markdown_paths
        return written

    def _write_window(self, start_step, end_step):
        window_decisions = [
            record for record in self.decisions
            if start_step < record.get('train_step', 0) <= end_step]
        window_episodes = [
            record for record in self.episodes
            if start_step <= record.get('train_step_end', 0) <= end_step]
        if not window_decisions and not window_episodes:
            return []

        report = self._build_report(start_step, end_step, window_decisions, window_episodes)
        stem = '{:08d}_{:08d}'.format(start_step, end_step)
        decisions_path = self.output_dir / ('decisions_{}.jsonl'.format(stem))
        episodes_path = self.output_dir / ('episodes_{}.jsonl'.format(stem))
        json_path = self.output_dir / ('evidence_window_{}.json'.format(stem))
        markdown_path = self.output_dir / ('evidence_window_{}.md'.format(stem))

        self._write_jsonl(decisions_path, window_decisions)
        self._write_jsonl(episodes_path, window_episodes)
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
        markdown_path.write_text(
            self._render_markdown(report), encoding='utf-8')

        self.decisions = [
            record for record in self.decisions
            if record.get('train_step', 0) > end_step]
        self.episodes = [
            record for record in self.episodes
            if record.get('train_step_end', 0) > end_step]
        self.previous_global_trend = report['global_training_trend']
        return [decisions_path, episodes_path, json_path, markdown_path]

    @staticmethod
    def _write_jsonl(path, records):
        with path.open('w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(
                    record, ensure_ascii=False, allow_nan=False) + '\n')

    @staticmethod
    def _mean(records, key):
        values = [
            record.get(key) for record in records
            if isinstance(record.get(key), (int, float))
            and math.isfinite(float(record.get(key)))]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _rate(records, key):
        values = [
            record.get(key) for record in records
            if isinstance(record.get(key), bool)
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _chosen_candidate(decision):
        for candidate in decision.get('candidates', []):
            if candidate.get('is_chosen'):
                return candidate
        return None

    def _decision_flags(self, decision):
        chosen = self._chosen_candidate(decision)
        if not chosen or chosen.get('capability_match') is None:
            return False, False
        chosen_match = chosen['capability_match']
        valid_matches = [
            candidate.get('capability_match')
            for candidate in decision.get('candidates', [])
            if candidate.get('is_valid')
            and candidate.get('capability_match') is not None]
        max_match = max(valid_matches) if valid_matches else chosen_match
        low = chosen_match < self.low_threshold
        missed = max_match - chosen_match > self.better_alternative_gap
        return low, missed

    def _decision_metrics(self, decisions):
        chosen_matches = []
        low_count = 0
        missed_count = 0
        considered = 0
        for decision in decisions:
            chosen = self._chosen_candidate(decision)
            if not chosen or chosen.get('capability_match') is None:
                continue
            considered += 1
            chosen_matches.append(chosen['capability_match'])
            low, missed = self._decision_flags(decision)
            low_count += int(low)
            missed_count += int(missed)
        return {
            'decision_count': len(decisions),
            'capability_decision_count': considered,
            'average_chosen_capability_match': (
                sum(chosen_matches) / len(chosen_matches) if chosen_matches else None),
            'low_capability_match_decision_rate': (
                low_count / considered if considered else None),
            'missed_better_alternative_rate': (
                missed_count / considered if considered else None),
        }

    def _episode_groups(self, episodes):
        successful = [
            episode for episode in episodes
            if episode.get('success')
            and isinstance(episode.get('final_makespan'), (int, float))]
        successful.sort(key=lambda episode: episode['final_makespan'])
        group_size = max(1, int(math.ceil(len(successful) * 0.2))) if successful else 0
        high_ids = {
            episode['episode_id'] for episode in successful[:group_size]}
        low_ids = {
            episode['episode_id'] for episode in episodes
            if not episode.get('success')}
        if group_size:
            low_ids.update(
                episode['episode_id'] for episode in successful[-group_size:])
        return high_ids, low_ids

    def _representative_cases(self, decisions, episode_ids, successful):
        cases = []
        for decision in decisions:
            if decision.get('episode_id') not in episode_ids:
                continue
            chosen = self._chosen_candidate(decision)
            if not chosen:
                continue
            low, missed = self._decision_flags(decision)
            chosen_match = chosen.get('capability_match')
            if successful:
                if chosen_match is None or chosen_match < self.low_threshold:
                    continue
                score = (
                    chosen_match,
                    chosen.get('model_prob') or 0)
            else:
                score = (int(missed), int(low), -(chosen_match or 0))
            cases.append((score, {
                'episode_id': decision.get('episode_id'),
                'decision_id': decision.get('decision_id'),
                'train_step': decision.get('train_step'),
                'current_agent_id': decision.get('current_agent_id'),
                'current_agent_skill_vector': decision.get('current_agent_skill_vector'),
                'current_open_task_count': decision.get('current_open_task_count'),
                'chosen_task_id': decision.get('chosen_task_id'),
                'final_outcome': {
                    'success': decision.get('eventual_episode_success'),
                    'deadlock': decision.get('eventual_deadlock'),
                    'makespan': decision.get('eventual_makespan'),
                },
                'decoder_logit_debug': decision.get('decoder_logit_debug'),
                'candidates': decision.get('candidates', []),
                'low_capability_match': low,
                'missed_better_capability_alternative': missed,
            }))
        cases.sort(key=lambda item: item[0], reverse=True)
        return [case for _, case in cases[:self.max_cases_per_report]]

    def _build_report(self, start_step, end_step, decisions, episodes):
        trend = {
            'episode_count': len(episodes),
            'decision_count': len(decisions),
            'success_rate': self._rate(episodes, 'success'),
            'deadlock_rate': self._rate(episodes, 'deadlock'),
            'timeout_rate': self._rate(episodes, 'timeout'),
            'average_makespan': self._mean(episodes, 'final_makespan'),
            'average_waiting_time': self._mean(episodes, 'average_waiting_time'),
            'average_wasted_ability_ratio': self._mean(
                episodes, 'average_wasted_ability_ratio'),
            'average_reward': self._mean(episodes, 'reward'),
            'average_open_task_count': self._mean(episodes, 'average_open_tasks'),
            'max_open_task_count': max(
                [episode.get('max_open_tasks') for episode in episodes
                 if isinstance(episode.get('max_open_tasks'), (int, float))],
                default=None),
        }
        changes = {}
        for key, value in trend.items():
            previous = (self.previous_global_trend or {}).get(key)
            changes[key] = (
                value - previous
                if isinstance(value, (int, float))
                and isinstance(previous, (int, float)) else None)

        low_count = 0
        missed_count = 0
        for decision in decisions:
            low, missed = self._decision_flags(decision)
            low_count += int(low)
            missed_count += int(missed)

        deadlock_tail_count = 0
        deadlock_ids = {
            episode['episode_id'] for episode in episodes if episode.get('deadlock')}
        for episode_id in deadlock_ids:
            episode_decisions = [
                decision for decision in decisions
                if decision.get('episode_id') == episode_id]
            for decision in episode_decisions[-self.deadlock_lookback:]:
                low, _ = self._decision_flags(decision)
                deadlock_tail_count += int(low)

        high_ids, low_ids = self._episode_groups(episodes)
        failed_ids = {
            episode['episode_id'] for episode in episodes
            if not episode.get('success')}
        high_decisions = [
            decision for decision in decisions
            if decision.get('episode_id') in high_ids]
        low_decisions = [
            decision for decision in decisions
            if decision.get('episode_id') in low_ids]
        high_metrics = self._decision_metrics(high_decisions)
        low_metrics = self._decision_metrics(low_decisions)
        failed_cases = self._representative_cases(
            decisions, failed_ids, successful=False)
        successful_cases = self._representative_cases(
            decisions, high_ids, successful=True)

        considered = self._decision_metrics(decisions)['capability_decision_count']
        failure_modes = {
            'low_capability_match_decisions': {
                'count': low_count,
                'rate': low_count / considered if considered else None,
            },
            'missed_better_capability_alternative': {
                'count': missed_count,
                'rate': missed_count / considered if considered else None,
            },
            'poor_capability_match_before_deadlock': {
                'count': deadlock_tail_count,
                'deadlock_episode_count': len(deadlock_ids),
                'lookback_decisions': self.deadlock_lookback,
            },
        }

        if missed_count:
            diagnosis_parts = [
                'The current policy sometimes selects tasks with low '
                'capability_match even when a higher-match valid alternative '
                'is present.']
            high_missed_rate = high_metrics.get('missed_better_alternative_rate')
            low_missed_rate = low_metrics.get('missed_better_alternative_rate')
            if (isinstance(high_missed_rate, (int, float))
                    and isinstance(low_missed_rate, (int, float))
                    and low_missed_rate > high_missed_rate):
                diagnosis_parts.append(
                    'This pattern is more frequent in failed or low-quality '
                    'episodes than in high-quality episodes.')
            if deadlock_tail_count:
                diagnosis_parts.append(
                    'Repeated poor-match choices also appear shortly before '
                    'deadlock in this window.')
            diagnosis_parts.append(
                'A future decoder bias may need to increase the weight of '
                'capability_match.')
            diagnosis = ' '.join(diagnosis_parts)
        elif low_count:
            diagnosis = (
                'The current policy makes some low capability_match choices, '
                'but the recorded candidates do not consistently show a '
                'higher-match valid alternative.')
        else:
            diagnosis = (
                'This window does not provide evidence of a recurring '
                'capability_match failure mode.')

        return {
            'window': {
                'start_step_exclusive': start_step,
                'end_step_inclusive': end_step,
            },
            'global_training_trend': trend,
            'change_from_previous_window': changes,
            'failure_mode_distribution': failure_modes,
            'feature_definition': {
                'name': 'capability_match',
                'type': 'binary',
                'definition': (
                    'Whether the current agent can reduce a candidate task '
                    'remaining requirement. It reuses '
                    'TaskEnv.compute_capability_match_from_existing_logic, '
                    'which is also used by the original contributable-task mask.'),
            },
            'explicit_bias_feature_definitions': {
                'completion_potential': (
                    '1 when the current agent can close the remaining task '
                    'requirement and make the coalition executable.'),
                'requirement_reduction_ratio': (
                    'Fraction of the task remaining requirement reduced by '
                    'the current agent, clipped to [0, 1].'),
                'travel_time': (
                    'Current-agent travel time to the task normalized by the '
                    'maximum direct map travel time; larger is farther.'),
                'waiting_pressure': (
                    'Normalized maximum waiting time among agents already '
                    'queued at the task coalition.'),
            },
            'aggregate_contrast': {
                'high_quality_decisions': high_metrics,
                'low_quality_or_failed_decisions': low_metrics,
                'high_quality_episode_ids': sorted(high_ids),
                'low_quality_or_failed_episode_ids': sorted(low_ids),
            },
            'representative_failed_decision_cases': failed_cases,
            'representative_successful_decision_cases': successful_cases,
            'current_policy_diagnosis': diagnosis,
            'recommended_llm_output_format': {
                'weights': zero_feature_weights(),
                'lambda': 0.0,
                'clip_range': [-2.0, 2.0],
                'rationale': {
                    'main_failure_modes': [],
                    'expected_effect': [],
                },
            },
        }

    @staticmethod
    def _format_value(value, percent=False):
        if value is None:
            return 'N/A'
        if percent:
            return '{:.2%}'.format(value)
        if isinstance(value, float):
            return '{:.4f}'.format(value)
        return str(value)

    @staticmethod
    def _format_vector(value, max_items=64):
        if value is None:
            return 'N/A'
        if not isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        rendered = []
        for item in value[:max_items]:
            if isinstance(item, float):
                rendered.append(round(item, 6))
            else:
                rendered.append(item)
        if len(value) > max_items:
            rendered.append('... {} more'.format(len(value) - max_items))
        return json.dumps(rendered, ensure_ascii=False)

    def _render_case(self, case, successful):
        debug = case.get('decoder_logit_debug') or {}
        lines = [
            '### Episode {} / Decision {}'.format(
                case.get('episode_id'), case.get('decision_id')),
            '',
            '**Context**',
            '',
            '- Agent: {}'.format(case.get('current_agent_id')),
            '- Agent skill vector: `{}`'.format(
                json.dumps(case.get('current_agent_skill_vector'))),
            '- Open task count: {}'.format(case.get('current_open_task_count')),
            '- Chosen task: {}'.format(case.get('chosen_task_id')),
            '- Final outcome: `{}`'.format(json.dumps(case.get('final_outcome'))),
            '',
            '**Candidate comparison**',
            '',
            '| Task | Probability | Raw logit | Feature bias | Biased logit | Chosen | Valid | State | Capability match | Explicit features | Remaining requirement |',
            '|---|---:|---:|---:|---:|:---:|:---:|---|---:|---|---|',
        ]
        for candidate in case.get('candidates', []):
            task_label = (
                'depot' if candidate.get('task_id') is None
                else 'm_{}'.format(candidate.get('task_id')))
            lines.append(
                '| {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` |'.format(
                    task_label,
                    self._format_value(candidate.get('model_prob')),
                    self._format_value(candidate.get('model_logit')),
                    self._format_value(
                        candidate.get('explicit_feature_logit_bias')),
                    self._format_value(candidate.get('biased_model_logit')),
                    'yes' if candidate.get('is_chosen') else 'no',
                    'yes' if candidate.get('is_valid') else 'no',
                    candidate.get('task_state'),
                    self._format_value(candidate.get('capability_match')),
                    json.dumps(
                        candidate.get('explicit_features', {}),
                        ensure_ascii=False),
                    json.dumps(candidate.get('remaining_requirement'))))
        if debug:
            lines.extend([
                '',
                '**Decoder logit debug**',
                '',
                '- Bias config: step={}, apply={}, weights=`{}`, lambda={}, clip_range=`{}`'.format(
                    debug.get('bias_global_step'),
                    debug.get('bias_apply'),
                    json.dumps(debug.get('used_weights'), ensure_ascii=False),
                    self._format_value(debug.get('used_lambda')),
                    json.dumps(debug.get('clip_range'))),
                '- Action order: {}'.format(debug.get('action_index_order')),
                '- Feature names: `{}`'.format(
                    self._format_vector(debug.get('feature_names'))),
                '- Valid action mask: `{}`'.format(
                    self._format_vector(debug.get('valid_action_mask'))),
                '- Capability match: `{}`'.format(
                    self._format_vector(debug.get('capability_match'))),
                '- Explicit features: `{}`'.format(
                    self._format_vector(debug.get('explicit_features'))),
                '- Raw decoder logits: `{}`'.format(
                    self._format_vector(debug.get('raw_decoder_logits'))),
                '- Explicit feature logit bias: `{}`'.format(
                    self._format_vector(
                        debug.get('explicit_feature_logit_bias'))),
                '- Biased decoder logits: `{}`'.format(
                    self._format_vector(debug.get('biased_decoder_logits'))),
            ])
        lines.extend(['', '**Interpretation**', ''])
        if successful:
            lines.append(
                'The chosen action had a strong capability match and appeared '
                'in a high-quality successful episode.')
        elif case.get('missed_better_capability_alternative'):
            lines.append(
                'The policy selected a low-match task while a higher-match '
                'valid candidate was available.')
        elif case.get('low_capability_match'):
            lines.append(
                'The selected task had low capability match in a low-quality '
                'or failed episode.')
        else:
            lines.append(
                'This is a representative decision from a low-quality or '
                'failed episode; capability match alone does not explain it.')
        lines.append('')
        return lines

    def _render_markdown(self, report):
        trend = report['global_training_trend']
        changes = report['change_from_previous_window']
        failure = report['failure_mode_distribution']
        contrast = report['aggregate_contrast']
        lines = [
            '# Evidence-Rich Training Report',
            '',
            'Window: ({}, {}]'.format(
                report['window']['start_step_exclusive'],
                report['window']['end_step_inclusive']),
            '',
            '## A. Global training trend',
            '',
        ]
        metric_specs = [
            ('Success rate', 'success_rate', True),
            ('Deadlock rate', 'deadlock_rate', True),
            ('Timeout rate', 'timeout_rate', True),
            ('Average makespan', 'average_makespan', False),
            ('Average waiting time', 'average_waiting_time', False),
            ('Average wasted ability ratio', 'average_wasted_ability_ratio', False),
            ('Average reward', 'average_reward', False),
            ('Average open task count', 'average_open_task_count', False),
            ('Maximum open task count', 'max_open_task_count', False),
        ]
        for label, key, percent in metric_specs:
            lines.append('- {}: {} (change: {})'.format(
                label,
                self._format_value(trend.get(key), percent=percent),
                self._format_value(changes.get(key), percent=percent)))

        lines.extend([
            '',
            '## B. Failure mode distribution',
            '',
            '- Low capability match decisions: {} ({})'.format(
                failure['low_capability_match_decisions']['count'],
                self._format_value(
                    failure['low_capability_match_decisions']['rate'], True)),
            '- Low-match choices with a better valid alternative: {} ({})'.format(
                failure['missed_better_capability_alternative']['count'],
                self._format_value(
                    failure['missed_better_capability_alternative']['rate'], True)),
            '- Poor-match decisions near deadlock: {} across {} deadlock episodes'.format(
                failure['poor_capability_match_before_deadlock']['count'],
                failure['poor_capability_match_before_deadlock'][
                    'deadlock_episode_count']),
            '',
            '## C. Feature definition',
            '',
            '`capability_match` measures whether the current deciding agent can '
            'reduce a candidate task remaining skill requirement. It reuses the '
            'project original ability-mask logic in '
            '`TaskEnv.compute_capability_match_from_existing_logic`, which is '
            'also called by `get_contributable_task_mask`.',
            '',
            'The current implementation is binary because the original project '
            'logic only checks whether the agent can reduce the remaining '
            'requirement.',
            '',
            'The LLM-controlled decoder bias uses four explicit action-level '
            'features: `completion_potential`, `requirement_reduction_ratio`, '
            '`travel_time`, and `waiting_pressure`. These are deterministic '
            'environment features computed for each candidate action before '
            'masking; depot and padding actions use zero feature values.',
            '',
            '## D. Aggregate contrast',
            '',
            '**High-quality decisions**',
            '',
        ])
        for key, value in contrast['high_quality_decisions'].items():
            lines.append('- {}: {}'.format(key, self._format_value(value)))
        lines.extend(['', '**Low-quality or failed decisions**', ''])
        for key, value in contrast['low_quality_or_failed_decisions'].items():
            lines.append('- {}: {}'.format(key, self._format_value(value)))

        lines.extend(['', '## E. Representative failed decision cases', ''])
        failed_cases = report['representative_failed_decision_cases']
        if failed_cases:
            for case in failed_cases:
                lines.extend(self._render_case(case, successful=False))
        else:
            lines.append('No failed cases were available in this window.')

        lines.extend(['', '## F. Representative successful decision cases', ''])
        successful_cases = report['representative_successful_decision_cases']
        if successful_cases:
            for case in successful_cases:
                lines.extend(self._render_case(case, successful=True))
        else:
            lines.append('No high-quality successful cases were available in this window.')

        lines.extend([
            '',
            '## G. Current policy diagnosis',
            '',
            report['current_policy_diagnosis'],
            '',
            '## H. Recommended LLM output format',
            '',
            '```json',
            json.dumps(
                report['recommended_llm_output_format'],
                indent=2,
                ensure_ascii=False),
            '```',
            '',
        ])
        return '\n'.join(lines)
