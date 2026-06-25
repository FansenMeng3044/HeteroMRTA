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

    @staticmethod
    def _feature_value(candidate, feature):
        features = candidate.get('explicit_features') or {}
        value = features.get(feature)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    @staticmethod
    def _rate_value(count, total):
        return count / total if total else None

    @staticmethod
    def _valid_task_candidates(decision):
        return [
            candidate for candidate in decision.get('candidates', [])
            if candidate.get('is_valid') and candidate.get('task_id') is not None
        ]

    def _best_candidate_feature(self, candidates, feature):
        values = [
            self._feature_value(candidate, feature) for candidate in candidates]
        values = [value for value in values if value is not None]
        if not values:
            return None
        if feature == 'travel_time':
            return min(values)
        return max(values)

    def _feature_means(self, decisions, best_valid=False):
        sums = zero_feature_weights()
        counts = {feature: 0 for feature in EXPLICIT_BIAS_FEATURES}
        for decision in decisions:
            if best_valid:
                candidates = self._valid_task_candidates(decision)
                if not candidates:
                    continue
            else:
                candidate = self._chosen_candidate(decision)
                if candidate is None:
                    continue
            for feature in EXPLICIT_BIAS_FEATURES:
                if best_valid:
                    value = self._best_candidate_feature(candidates, feature)
                else:
                    value = self._feature_value(candidate, feature)
                if value is None:
                    continue
                sums[feature] += value
                counts[feature] += 1
        return {
            feature: (sums[feature] / counts[feature]
                      if counts[feature] else None)
            for feature in EXPLICIT_BIAS_FEATURES
        }

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

    def _explicit_feature_signal_metrics(self, decisions, high_ids, low_ids):
        thresholds = {
            'completion_potential_high': 0.5,
            'completion_gap': 0.5,
            'requirement_reduction_gap': 0.4,
            'travel_time_gap': 0.25,
            'waiting_pressure_gap': 0.25,
        }
        total_decisions = len(decisions)
        decisions_with_candidates = 0
        depot_choice_count = 0
        depot_with_completion_count = 0
        missed_completion_count = 0
        missed_reduction_count = 0
        long_travel_count = 0
        missed_waiting_count = 0

        for decision in decisions:
            chosen = self._chosen_candidate(decision)
            if chosen is None:
                continue
            valid_tasks = self._valid_task_candidates(decision)
            if not valid_tasks:
                continue
            decisions_with_candidates += 1
            chosen_is_depot = (
                chosen.get('task_id') is None or chosen.get('action_index') == 0)
            if chosen_is_depot:
                depot_choice_count += 1

            best_completion = self._best_candidate_feature(
                valid_tasks, 'completion_potential')
            best_reduction = self._best_candidate_feature(
                valid_tasks, 'requirement_reduction_ratio')
            nearest_travel = self._best_candidate_feature(
                valid_tasks, 'travel_time')
            best_waiting = self._best_candidate_feature(
                valid_tasks, 'waiting_pressure')

            chosen_completion = self._feature_value(
                chosen, 'completion_potential')
            chosen_reduction = self._feature_value(
                chosen, 'requirement_reduction_ratio')
            chosen_travel = self._feature_value(chosen, 'travel_time')
            chosen_waiting = self._feature_value(chosen, 'waiting_pressure')

            if chosen_is_depot:
                chosen_completion = 0.0 if chosen_completion is None else chosen_completion
                chosen_reduction = 0.0 if chosen_reduction is None else chosen_reduction
                chosen_waiting = 0.0 if chosen_waiting is None else chosen_waiting

            if (best_completion is not None
                    and best_completion >= thresholds['completion_potential_high']):
                if chosen_is_depot:
                    depot_with_completion_count += 1
                if (chosen_completion is not None
                        and best_completion - chosen_completion
                        >= thresholds['completion_gap']):
                    missed_completion_count += 1

            if (best_reduction is not None and chosen_reduction is not None
                    and best_reduction - chosen_reduction
                    >= thresholds['requirement_reduction_gap']):
                missed_reduction_count += 1

            if (not chosen_is_depot and nearest_travel is not None
                    and chosen_travel is not None
                    and chosen_travel - nearest_travel
                    >= thresholds['travel_time_gap']):
                long_travel_count += 1

            if (best_waiting is not None and chosen_waiting is not None
                    and best_waiting - chosen_waiting
                    >= thresholds['waiting_pressure_gap']):
                missed_waiting_count += 1

        high_decisions = [
            decision for decision in decisions
            if decision.get('episode_id') in high_ids]
        low_decisions = [
            decision for decision in decisions
            if decision.get('episode_id') in low_ids]
        return {
            'decision_count': total_decisions,
            'decisions_with_valid_task_candidates': decisions_with_candidates,
            'depot_choice': {
                'count': depot_choice_count,
                'rate': self._rate_value(depot_choice_count, total_decisions),
            },
            'depot_with_completion_candidate': {
                'count': depot_with_completion_count,
                'rate_among_depot_choices': self._rate_value(
                    depot_with_completion_count, depot_choice_count),
                'rate_among_all_decisions': self._rate_value(
                    depot_with_completion_count, total_decisions),
            },
            'missed_completion_potential': {
                'count': missed_completion_count,
                'rate': self._rate_value(
                    missed_completion_count, decisions_with_candidates),
            },
            'missed_requirement_reduction': {
                'count': missed_reduction_count,
                'rate': self._rate_value(
                    missed_reduction_count, decisions_with_candidates),
            },
            'long_travel_choice': {
                'count': long_travel_count,
                'rate': self._rate_value(
                    long_travel_count, decisions_with_candidates),
            },
            'missed_waiting_pressure': {
                'count': missed_waiting_count,
                'rate': self._rate_value(
                    missed_waiting_count, decisions_with_candidates),
            },
            'chosen_feature_means': {
                'all_decisions': self._feature_means(decisions),
                'high_quality_decisions': self._feature_means(high_decisions),
                'low_quality_or_failed_decisions': self._feature_means(
                    low_decisions),
            },
            'best_valid_feature_means': {
                'all_decisions': self._feature_means(
                    decisions, best_valid=True),
            },
            'thresholds': thresholds,
        }

    def _time_quality_signal_metrics(self, decisions, episodes):
        thresholds = {
            'makespan_gap_min': 1.0,
            'makespan_relative_gap': 0.05,
            'feature_mean_gap': 0.05,
        }
        successful = [
            episode for episode in episodes
            if episode.get('success')
            and isinstance(episode.get('final_makespan'), (int, float))
            and math.isfinite(float(episode.get('final_makespan')))]
        successful.sort(key=lambda episode: episode['final_makespan'])
        group_size = (
            max(1, int(math.ceil(len(successful) * 0.2)))
            if successful else 0)
        fastest = successful[:group_size]
        slowest = successful[-group_size:] if group_size else []
        fastest_ids = {episode['episode_id'] for episode in fastest}
        slowest_ids = {episode['episode_id'] for episode in slowest}
        fastest_decisions = [
            decision for decision in decisions
            if decision.get('episode_id') in fastest_ids]
        slowest_decisions = [
            decision for decision in decisions
            if decision.get('episode_id') in slowest_ids]
        fastest_features = self._feature_means(fastest_decisions)
        slowest_features = self._feature_means(slowest_decisions)
        feature_gap = {}
        for feature in EXPLICIT_BIAS_FEATURES:
            fast_value = fastest_features.get(feature)
            slow_value = slowest_features.get(feature)
            feature_gap[feature] = (
                slow_value - fast_value
                if isinstance(fast_value, (int, float))
                and isinstance(slow_value, (int, float)) else None)

        fastest_makespan = self._mean(fastest, 'final_makespan')
        slowest_makespan = self._mean(slowest, 'final_makespan')
        fastest_waiting = self._mean(fastest, 'average_waiting_time')
        slowest_waiting = self._mean(slowest, 'average_waiting_time')
        makespan_gap = (
            slowest_makespan - fastest_makespan
            if isinstance(fastest_makespan, (int, float))
            and isinstance(slowest_makespan, (int, float)) else None)
        relative_gap = (
            makespan_gap / max(abs(fastest_makespan), 1e-9)
            if isinstance(makespan_gap, (int, float))
            and isinstance(fastest_makespan, (int, float)) else None)
        waiting_gap = (
            slowest_waiting - fastest_waiting
            if isinstance(fastest_waiting, (int, float))
            and isinstance(slowest_waiting, (int, float)) else None)
        time_optimization_needed = bool(
            len(successful) >= 2
            and isinstance(makespan_gap, (int, float))
            and isinstance(relative_gap, (int, float))
            and makespan_gap >= thresholds['makespan_gap_min']
            and relative_gap >= thresholds['makespan_relative_gap'])

        directions = {}
        gap_threshold = thresholds['feature_mean_gap']
        if time_optimization_needed:
            travel_gap = feature_gap.get('travel_time')
            if isinstance(travel_gap, (int, float)) and travel_gap >= gap_threshold:
                directions['travel_time'] = 'negative'
            completion_gap = feature_gap.get('completion_potential')
            if (isinstance(completion_gap, (int, float))
                    and completion_gap <= -gap_threshold):
                directions['completion_potential'] = 'positive'
            reduction_gap = feature_gap.get('requirement_reduction_ratio')
            if (isinstance(reduction_gap, (int, float))
                    and reduction_gap <= -gap_threshold):
                directions['requirement_reduction_ratio'] = 'positive'
            waiting_gap_feature = feature_gap.get('waiting_pressure')
            if (isinstance(waiting_gap_feature, (int, float))
                    and waiting_gap_feature <= -gap_threshold):
                directions['waiting_pressure'] = 'positive'

        return {
            'objective': 'minimize successful-episode makespan and waiting time, not only maximize success rate',
            'successful_episode_count': len(successful),
            'all_recorded_episodes_successful': bool(
                episodes and len(successful) == len(episodes)),
            'group_size': group_size,
            'fastest_successful_episode_ids': sorted(fastest_ids),
            'slowest_successful_episode_ids': sorted(slowest_ids),
            'fastest_successful_metrics': {
                'average_makespan': fastest_makespan,
                'average_waiting_time': fastest_waiting,
                'average_reward': self._mean(fastest, 'reward'),
            },
            'slowest_successful_metrics': {
                'average_makespan': slowest_makespan,
                'average_waiting_time': slowest_waiting,
                'average_reward': self._mean(slowest, 'reward'),
            },
            'slow_minus_fast': {
                'makespan': makespan_gap,
                'makespan_relative': relative_gap,
                'average_waiting_time': waiting_gap,
                'chosen_feature_means': feature_gap,
            },
            'fastest_chosen_feature_means': fastest_features,
            'slowest_chosen_feature_means': slowest_features,
            'feature_weight_directions': directions,
            'time_optimization_needed': time_optimization_needed,
            'thresholds': thresholds,
        }

    def _llm_bias_guidance(self, trend, explicit_signal, time_quality_signal):
        suggested_weights = zero_feature_weights()
        main_failure_modes = []
        expected_effect = []

        completion_count = (
            explicit_signal['depot_with_completion_candidate']['count']
            + explicit_signal['missed_completion_potential']['count'])
        if completion_count:
            suggested_weights['completion_potential'] = 0.8
            suggested_weights['requirement_reduction_ratio'] = max(
                suggested_weights['requirement_reduction_ratio'], 0.4)
            main_failure_modes.append(
                'completion-ready valid tasks were ignored or depot was selected while one was available')
            expected_effect.append(
                'increase logits for actions that can immediately close a task requirement')

        if explicit_signal['missed_requirement_reduction']['count']:
            suggested_weights['requirement_reduction_ratio'] = max(
                suggested_weights['requirement_reduction_ratio'], 0.6)
            main_failure_modes.append(
                'chosen actions often reduced less remaining requirement than another valid candidate')
            expected_effect.append(
                'prefer actions that reduce more of the remaining task requirement')

        if explicit_signal['long_travel_choice']['count']:
            suggested_weights['travel_time'] = min(
                suggested_weights['travel_time'], -0.5)
            main_failure_modes.append(
                'chosen task actions were farther than another valid candidate')
            expected_effect.append(
                'penalize unnecessarily long travel among otherwise valid tasks')

        if explicit_signal['missed_waiting_pressure']['count']:
            suggested_weights['waiting_pressure'] = max(
                suggested_weights['waiting_pressure'], 0.3)
            main_failure_modes.append(
                'chosen actions ignored candidates with higher waiting pressure')
            expected_effect.append(
                'prioritize tasks where agents are already waiting for coalition completion')

        if time_quality_signal.get('time_optimization_needed'):
            gap = time_quality_signal.get('slow_minus_fast', {})
            main_failure_modes.append(
                'success rate can be high while slow successful episodes still have larger makespan')
            expected_effect.append(
                'optimize for shorter makespan and lower waiting time even when all tasks finish')
            directions = time_quality_signal.get('feature_weight_directions', {})
            if directions.get('travel_time') == 'negative':
                suggested_weights['travel_time'] = min(
                    suggested_weights['travel_time'], -0.6)
                expected_effect.append(
                    'make slow high-travel successful decisions less likely')
            if directions.get('completion_potential') == 'positive':
                suggested_weights['completion_potential'] = max(
                    suggested_weights['completion_potential'], 0.6)
            if directions.get('requirement_reduction_ratio') == 'positive':
                suggested_weights['requirement_reduction_ratio'] = max(
                    suggested_weights['requirement_reduction_ratio'], 0.5)
            if directions.get('waiting_pressure') == 'positive':
                suggested_weights['waiting_pressure'] = max(
                    suggested_weights['waiting_pressure'], 0.3)
            if not directions and not any(
                    value != 0.0 for value in suggested_weights.values()):
                suggested_weights['travel_time'] = -0.3
                expected_effect.append(
                    'conservatively favor shorter-travel choices to test makespan improvement')
            main_failure_modes.append(
                'slow-minus-fast makespan gap is {}'.format(
                    self._format_value(gap.get('makespan'))))

        should_return_nonzero = any(
            value != 0.0 for value in suggested_weights.values())
        return {
            'should_return_nonzero_bias': should_return_nonzero,
            'suggested_weights': suggested_weights,
            'suggested_lambda': 0.25 if should_return_nonzero else 0.0,
            'suggested_clip_range': [-2.0, 2.0],
            'main_failure_modes': main_failure_modes,
            'expected_effect': expected_effect,
            'success_rate': trend.get('success_rate'),
            'timeout_rate': trend.get('timeout_rate'),
            'deadlock_rate': trend.get('deadlock_rate'),
            'time_optimization_needed': time_quality_signal.get(
                'time_optimization_needed'),
        }

    def _diagnose_policy(self, missed_count, low_count, deadlock_tail_count,
                         high_metrics, low_metrics, explicit_signal,
                         time_quality_signal, llm_guidance):
        diagnosis_parts = []
        if missed_count:
            diagnosis_parts.append(
                'The capability diagnostic still finds choices with lower '
                'capability_match than another valid alternative.')
            high_missed_rate = high_metrics.get('missed_better_alternative_rate')
            low_missed_rate = low_metrics.get('missed_better_alternative_rate')
            if (isinstance(high_missed_rate, (int, float))
                    and isinstance(low_missed_rate, (int, float))
                    and low_missed_rate > high_missed_rate):
                diagnosis_parts.append(
                    'This capability pattern is more frequent in failed or '
                    'low-quality episodes than in high-quality episodes.')
            if deadlock_tail_count:
                diagnosis_parts.append(
                    'Repeated poor-match choices also appear shortly before '
                    'deadlock in this window.')
        elif low_count:
            diagnosis_parts.append(
                'The capability diagnostic sees some low capability_match '
                'choices, but not a consistent higher-match valid alternative.')
        else:
            diagnosis_parts.append(
                'Capability_match is not a useful failure discriminator in '
                'this window because valid task candidates are already '
                'contributable under the mask.')

        if time_quality_signal.get('time_optimization_needed'):
            gap = time_quality_signal.get('slow_minus_fast', {})
            diagnosis_parts.append(
                'Success alone is not the target: slow successful episodes are '
                'still slower than fast successful episodes by makespan {} '
                'and waiting time {}.'.format(
                    self._format_value(gap.get('makespan')),
                    self._format_value(gap.get('average_waiting_time'))))

        if llm_guidance['should_return_nonzero_bias']:
            signal_parts = []
            for key, label in (
                    ('depot_with_completion_candidate',
                     'depot choices while completion-ready tasks existed'),
                    ('missed_completion_potential',
                     'missed completion-potential alternatives'),
                    ('missed_requirement_reduction',
                     'missed higher requirement-reduction alternatives'),
                    ('long_travel_choice', 'unnecessarily long-travel choices'),
                    ('missed_waiting_pressure',
                     'missed higher waiting-pressure alternatives')):
                count = explicit_signal[key]['count']
                if count:
                    signal_parts.append('{} {}'.format(count, label))
            diagnosis_parts.append(
                'Explicit decoder-bias features do show actionable signals: '
                + '; '.join(signal_parts) + '.')
            diagnosis_parts.append(
                'DeepSeek should consider non-zero feature weights with '
                'lambda > 0 instead of returning the all-zero no-op config.')
        else:
            diagnosis_parts.append(
                'The explicit feature aggregate does not show a strong '
                'actionable bias direction in this window, so an all-zero '
                'bias may be reasonable.')
        return ' '.join(diagnosis_parts)

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
            decisions, low_ids, successful=False)
        successful_cases = self._representative_cases(
            decisions, high_ids, successful=True)
        explicit_feature_signal = self._explicit_feature_signal_metrics(
            decisions, high_ids, low_ids)
        time_quality_signal = self._time_quality_signal_metrics(
            decisions, episodes)
        llm_bias_guidance = self._llm_bias_guidance(
            trend, explicit_feature_signal, time_quality_signal)

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

        diagnosis = self._diagnose_policy(
            missed_count,
            low_count,
            deadlock_tail_count,
            high_metrics,
            low_metrics,
            explicit_feature_signal,
            time_quality_signal,
            llm_bias_guidance)

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
            'explicit_feature_failure_signal': explicit_feature_signal,
            'time_quality_signal': time_quality_signal,
            'llm_bias_guidance': llm_bias_guidance,
            'representative_failed_decision_cases': failed_cases,
            'representative_successful_decision_cases': successful_cases,
            'current_policy_diagnosis': diagnosis,
            'recommended_llm_output_format': {
                'weights': llm_bias_guidance['suggested_weights'],
                'lambda': llm_bias_guidance['suggested_lambda'],
                'clip_range': llm_bias_guidance['suggested_clip_range'],
                'rationale': {
                    'main_failure_modes': llm_bias_guidance[
                        'main_failure_modes'],
                    'expected_effect': llm_bias_guidance['expected_effect'],
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

        signal = report.get('explicit_feature_failure_signal') or {}
        time_signal = report.get('time_quality_signal') or {}
        guidance = report.get('llm_bias_guidance') or {}
        lines.extend(['', '## E. Explicit feature failure and time-quality signals', ''])
        if signal:
            depot = signal.get('depot_choice', {})
            depot_completion = signal.get(
                'depot_with_completion_candidate', {})
            missed_completion = signal.get('missed_completion_potential', {})
            missed_reduction = signal.get(
                'missed_requirement_reduction', {})
            long_travel = signal.get('long_travel_choice', {})
            missed_waiting = signal.get('missed_waiting_pressure', {})
            lines.extend([
                '- Decisions with valid task candidates: {} / {}'.format(
                    signal.get('decisions_with_valid_task_candidates'),
                    signal.get('decision_count')),
                '- Depot choices: {} ({})'.format(
                    depot.get('count'),
                    self._format_value(depot.get('rate'), True)),
                '- Depot despite completion-ready candidate: {} ({} of depot choices, {} of all decisions)'.format(
                    depot_completion.get('count'),
                    self._format_value(
                        depot_completion.get('rate_among_depot_choices'), True),
                    self._format_value(
                        depot_completion.get('rate_among_all_decisions'), True)),
                '- Missed completion-potential alternatives: {} ({})'.format(
                    missed_completion.get('count'),
                    self._format_value(missed_completion.get('rate'), True)),
                '- Missed higher requirement-reduction alternatives: {} ({})'.format(
                    missed_reduction.get('count'),
                    self._format_value(missed_reduction.get('rate'), True)),
                '- Unnecessarily long-travel choices: {} ({})'.format(
                    long_travel.get('count'),
                    self._format_value(long_travel.get('rate'), True)),
                '- Missed higher waiting-pressure alternatives: {} ({})'.format(
                    missed_waiting.get('count'),
                    self._format_value(missed_waiting.get('rate'), True)),
                '',
                '**Chosen feature means**',
                '',
            ])
            chosen_means = signal.get('chosen_feature_means', {})
            for label, key in (
                    ('All decisions', 'all_decisions'),
                    ('High-quality decisions', 'high_quality_decisions'),
                    ('Low-quality or failed decisions',
                     'low_quality_or_failed_decisions')):
                lines.append('- {}: `{}`'.format(
                    label,
                    json.dumps(chosen_means.get(key), ensure_ascii=False)))
            lines.extend([
                '- Best valid candidate means: `{}`'.format(
                    json.dumps(
                        signal.get('best_valid_feature_means', {}).get(
                            'all_decisions'),
                        ensure_ascii=False)),
            ])
            if time_signal:
                fastest = time_signal.get('fastest_successful_metrics', {})
                slowest = time_signal.get('slowest_successful_metrics', {})
                gap = time_signal.get('slow_minus_fast', {})
                lines.extend([
                    '',
                    '**Successful-episode time-quality signal**',
                    '',
                    '- Objective: {}'.format(time_signal.get('objective')),
                    '- Successful episodes: {}'.format(
                        time_signal.get('successful_episode_count')),
                    '- All recorded episodes successful: {}'.format(
                        'yes' if time_signal.get(
                            'all_recorded_episodes_successful') else 'no'),
                    '- Fastest successful episode IDs: `{}`'.format(
                        json.dumps(
                            time_signal.get('fastest_successful_episode_ids'),
                            ensure_ascii=False)),
                    '- Slowest successful episode IDs: `{}`'.format(
                        json.dumps(
                            time_signal.get('slowest_successful_episode_ids'),
                            ensure_ascii=False)),
                    '- Fastest average makespan / waiting: {} / {}'.format(
                        self._format_value(fastest.get('average_makespan')),
                        self._format_value(
                            fastest.get('average_waiting_time'))),
                    '- Slowest average makespan / waiting: {} / {}'.format(
                        self._format_value(slowest.get('average_makespan')),
                        self._format_value(
                            slowest.get('average_waiting_time'))),
                    '- Slow-minus-fast makespan / waiting: {} / {}'.format(
                        self._format_value(gap.get('makespan')),
                        self._format_value(gap.get('average_waiting_time'))),
                    '- Slow-minus-fast chosen feature means: `{}`'.format(
                        json.dumps(
                            gap.get('chosen_feature_means'),
                            ensure_ascii=False)),
                    '- Time optimization needed: {}'.format(
                        'yes' if time_signal.get(
                            'time_optimization_needed') else 'no'),
                    '- Feature weight directions: `{}`'.format(
                        json.dumps(
                            time_signal.get('feature_weight_directions'),
                            ensure_ascii=False)),
                ])
            lines.extend([
                '',
                '**LLM bias guidance**',
                '',
                '- Should return nonzero bias: {}'.format(
                    'yes' if guidance.get('should_return_nonzero_bias') else 'no'),
                '- Suggested weights: `{}`'.format(
                    json.dumps(
                        guidance.get('suggested_weights'),
                        ensure_ascii=False)),
                '- Suggested lambda: {}'.format(
                    self._format_value(guidance.get('suggested_lambda'))),
                '- Main failure modes: `{}`'.format(
                    json.dumps(
                        guidance.get('main_failure_modes'),
                        ensure_ascii=False)),
                '- Expected effect: `{}`'.format(
                    json.dumps(
                        guidance.get('expected_effect'),
                        ensure_ascii=False)),
            ])
        else:
            lines.append('No explicit feature signal summary was available.')

        lines.extend(['', '## F. Representative low-quality or failed decision cases', ''])
        failed_cases = report['representative_failed_decision_cases']
        if failed_cases:
            for case in failed_cases:
                lines.extend(self._render_case(case, successful=False))
        else:
            lines.append('No low-quality or failed cases were available in this window.')

        lines.extend(['', '## G. Representative successful decision cases', ''])
        successful_cases = report['representative_successful_decision_cases']
        if successful_cases:
            for case in successful_cases:
                lines.extend(self._render_case(case, successful=True))
        else:
            lines.append('No high-quality successful cases were available in this window.')

        lines.extend([
            '',
            '## H. Current policy diagnosis',
            '',
            report['current_policy_diagnosis'],
            '',
            '## I. Recommended LLM output format',
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
