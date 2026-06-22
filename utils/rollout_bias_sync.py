def bias_snapshot_version(snapshot):
    if not isinstance(snapshot, dict):
        return 0
    return int(snapshot.get('global_step') or 0)


def rollout_uses_active_snapshot(info, active_snapshot):
    if not isinstance(info, dict):
        return False
    return int(info.get('bias_global_step') or 0) == bias_snapshot_version(
        active_snapshot)


def slice_transition_buffer(buffer, transition_count):
    transition_count = max(0, int(transition_count))
    return {
        key: values[:transition_count]
        for key, values in buffer.items()
    }


def truncate_evidence_payloads(payloads, decision_count):
    """Keep evidence in rollout order up to an exact transition boundary."""
    remaining = max(0, int(decision_count))
    truncated = []
    kept = 0
    for payload in payloads:
        decisions = list(payload.get('decisions', []))
        if not decisions:
            if remaining > 0:
                truncated.append({
                    'episode': payload.get('episode', {}),
                    'decisions': [],
                })
            continue
        if remaining <= 0:
            break
        take = min(len(decisions), remaining)
        truncated.append({
            'episode': payload.get('episode', {}),
            'decisions': decisions[:take],
        })
        kept += take
        remaining -= take
        if take < len(decisions):
            break
    return truncated, kept


def complete_evidence_payloads(payloads, decision_count, episode_number):
    """Truncate to the boundary and fill missing optional evidence with nulls."""
    completed, kept = truncate_evidence_payloads(payloads, decision_count)
    missing = max(0, int(decision_count) - kept)
    if not missing:
        return completed

    episode_id = -abs(int(episode_number) + 1)
    decisions = [{
        'episode_id': episode_id,
        'decision_id': index,
        'current_agent_id': None,
        'current_agent_species': None,
        'current_agent_skill_vector': None,
        'current_open_task_count': None,
        'current_completed_task_count': None,
        'remaining_task_count': None,
        'chosen_task_id': None,
        'chosen_task_state': None,
        'model_entropy': None,
        'valid_action_count': None,
        'eventual_episode_success': None,
        'eventual_deadlock': None,
        'eventual_makespan': None,
        'eventual_awt': None,
        'eventual_awar': None,
        'candidates': [],
        'evidence_missing': True,
    } for index in range(missing)]
    completed.append({
        'episode': {
            'episode_id': episode_id,
            'success': None,
            'deadlock': None,
            'timeout': None,
            'final_makespan': None,
            'average_waiting_time': None,
            'average_wasted_ability_ratio': None,
            'reward': None,
            'num_agents': None,
            'num_tasks': None,
            'num_species': None,
            'num_skills': None,
            'task_to_agent_ratio': None,
            'max_open_tasks': None,
            'average_open_tasks': None,
            'deadlock_step': None,
            'num_decisions': missing,
            'evidence_missing': True,
        },
        'decisions': decisions,
    })
    return completed
