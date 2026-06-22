# DeepSeek Window-Level Capability Bias Specification

## Required Sequence

The training process uses 3000 effective transitions per evidence window:

```text
train for one 3000-transition window
generate the evidence report
call DeepSeek once from the main process
parse and sanitize the JSON response
update the cached capability_match bias with EMA
discard any late Ray result produced by an older snapshot
use the new immutable snapshot for all accepted rollout work
repeat at the next report boundary
```

DeepSeek must never be called from a model, policy, attention, decoder, or
forward method. Ray workers never own a DeepSeek client and never write
DeepSeek response files.

Each Ray result carries its bias version. If a window update succeeds while
older jobs are still in flight, their late results are discarded and those
actors are resubmitted with the active snapshot. The job that reaches a
boundary is accepted only up to the exact boundary; any overflow generated
with the old snapshot is discarded.

## Configuration

The Python configuration must provide defaults equivalent to:

```yaml
enable_evidence_logging: true
evidence_log_interval_steps: 3000
evidence_output_dir: "./evidence_logs"
max_cases_per_report: 20
max_candidates_per_decision: 8

enable_deepseek_bias: true
deepseek_bias_update_interval_steps: 3000
deepseek_base_url: "https://api.deepseek.com"
deepseek_model: "deepseek-v4-flash"
deepseek_temperature: 0.0
deepseek_max_tokens: 1024
deepseek_timeout: 60
deepseek_use_json_response: true
deepseek_response_output_dir: "./evidence_logs/deepseek_responses"

llm_bias_ema_alpha: 0.3
```

The API key is read only from `DEEPSEEK_API_KEY`. It must never be stored in
configuration, prompts, responses, logs, exceptions, or checkpoints.

## API Frequency And Recovery

- A report window can be claimed for at most one API attempt.
- The attempt marker is written before the network request, so a restart
  cannot repeat an uncertain request.
- A successful response writes prompt, raw response, sanitized response, and
  active bias files.
- Missing keys, timeout, transport failure, malformed JSON, unknown feature
  weights, non-finite numbers, or invalid ranges must not stop training.
- Before the first successful response the bias remains disabled.
- After a successful response, later failures retain the previous safe
  snapshot.
- Existing attempt and active-bias files are restored on restart.

## Sanitized Bias

Only this response schema is accepted:

```json
{
  "weights": {"capability_match": 0.0},
  "lambda": 0.0,
  "clip_range": [-2.0, 2.0],
  "rationale": {
    "main_failure_modes": [],
    "expected_effect": []
  }
}
```

The sanitized weight is clamped to `[-2, 2]`, lambda to `[0, 1]`, and clip
bounds to `[-10, 10]` with `low < high`. EMA is applied independently:

```text
used = alpha * sanitized_raw + (1 - alpha) * previous_used
```

The initial previous values are zero. Therefore an initial raw weight of
`0.8` produces a used weight of `0.24` when alpha is `0.3`.

The active file records global step, source report, apply flag, used and raw
values, EMA alpha, clip range, and update interval.

## Decoder And Training Consistency

Workers receive only a JSON-safe immutable snapshot. For each decision they
compute the action-space capability vector by reusing
`TaskEnv.get_capability_match`; depot and padded actions are zero.

The only explicit feature is `capability_match`. The bias is:

```text
clip(lambda * weight * capability_match, clip_range)
```

Invalid actions receive zero bias and retain the original mask. Bias is added
to pointer logits before masking and softmax.

Evidence records the raw decoder score before this bias is added, while action
probabilities come from the final biased and masked logits.
Each decision JSONL record includes `decoder_logit_debug`, which stores
`raw_decoder_logits`, `capability_logit_bias`, and `biased_decoder_logits`
for the action vector before invalid-action masking. Candidate records also
include per-action `capability_logit_bias` and `biased_model_logit`.
Representative cases in each window JSON and Markdown report carry the same
debug block so the report can be inspected without opening the decisions
JSONL file.

Each training transition stores both its capability vector and the exact bias
snapshot used during sampling. REINFORCE recomputation must use those stored
values, even after the active snapshot changes.

When DeepSeek bias is disabled or the active snapshot is disabled, the model
must follow the original no-bias code path without changing logits, masks,
sampling order, rewards, buffers, or trajectories.

## Files

```text
evidence_logs/
  evidence_window_00000000_00003000.md
  evidence_window_00000000_00003000.json
  deepseek_responses/
    deepseek_prompt_00000000_00003000.txt
    deepseek_raw_00000000_00003000.json
    deepseek_sanitized_00000000_00003000.json
    deepseek_attempt_00000000_00003000.json
  bias_configs/
    active_bias_config_00003000.json
```

A forced shutdown flush may write a partial report. On restart, its JSONL
records are reloaded and the same fixed window is continued to the next
3000-step boundary; partial flushes do not shift later window numbering.

## Acceptance Evidence

Tests must cover one call per report, persistent deduplication, initial-off
behavior, EMA, sanitization, malformed output, timeout fallback, environment
key handling, response files, snapshot restore, worker synchronization,
rollout/training probability consistency, unchanged masks, disabled trajectory
equivalence, 3000-step naming, and absence of network code in forward methods.
