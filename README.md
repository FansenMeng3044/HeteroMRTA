# HeteroMRTA
Code for RAL Paper: Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning.

This is a repository using deep reinforcement learning to address single-task agent (ST) multi-robot task(MR) task assignment problem.
We train agents make decisions sequentially, and then they are able to choose task in a decentralized manner in execution.

## Demo

<img src="env/demo.gif" alt="demo" style="width: 70%;">

## Code structure

Three main structures of the code are as below:
1. Environments: generate random tasks locations/ requirements and agents with their depot.
1. Neural network: network based on attention in Pytorch 
1. Ray framework: REINFORCE algorithm implementation in ray.

## Running instructions
1. Set hyperparameters in parameters.py then run ```python driver.py```
2. Testing the trained model by running ```python test.py```

1. requirements: 
    1. python => 3.6
    1. torch >= 1.8.1
    1. numpy, ray, matplotlib, scipy, pandas

## Evidence-rich training reports

Evidence logging is configured by `EvidenceParams` in `parameters.py`. The
recorder and optional DeepSeek bias use only the binary `capability_match`
feature and reuse the environment's original contributable-task mask logic.

Ray workers collect lightweight episode payloads. Only the main process in
`driver.py` writes files. At each configured transition window it creates:

```text
evidence_logs/
  decisions_<start>_<end>.jsonl
  episodes_<start>_<end>.jsonl
  evidence_window_<start>_<end>.json
  evidence_window_<start>_<end>.md
```

The default window is 3000 effective training transitions. After a complete
report is written, the main process may call DeepSeek once and cache the
sanitized bias for subsequent worker jobs. Late Ray results carrying an older
bias version are discarded, so accepted transitions in the next window use
the synchronized snapshot. Model and decoder forward methods never perform
network requests. The first window runs with zero bias, and
`enable_deepseek_bias=False` preserves the original logits.

The API key is read only from `DEEPSEEK_API_KEY`. DeepSeek prompts, raw
responses, sanitized responses, and active bias snapshots are stored under:

```text
evidence_logs/deepseek_responses/
evidence_logs/bias_configs/
```

See `docs/deepseek_bias_spec.md` for the update formula, EMA behavior,
failure fallback, restart deduplication, and file naming contract.

Run the focused verification suite with:

```text
python -m unittest discover -s tests -v
```

The suite checks mask equivalence, optional decoder logits, JSON
serialization, window output, report sections, disabled logging, candidate
limits, depot handling, timeout/deadlock classification, and fixed-seed
trajectory equivalence with logging enabled and disabled.
