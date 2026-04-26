# MLSYS Project

This repository is the working codebase for our MLSYS project on KV-cache compression for long-context LLM inference.

## References
- Project doc: https://docs.google.com/document/d/1pA3P6c7hVDhUGY-CWLCsV5OVtFPnspJNayaRbbXSPTY/edit?tab=t.0
- KVPress: https://github.com/NVIDIA/kvpress

## Scope
We are studying tradeoffs between:
- accuracy on long-context benchmarks
- KV-cache memory footprint
- decode-time efficiency

Current evaluation focus:
- LongBench
- additional targeted validations / ablations

## Environment Setup
Status: `done` for the initial baseline environment.

Typical local setup:

```bash
conda create -n snapkv-exp python=3.10 -y
conda activate snapkv-exp
pip install -e .
pip install "transformers==4.37.0" "datasets==2.19.2"
```

## Modal Setup
The repository includes a Modal-based experiment runner in [`test.py`](./test.py).

Install and authenticate Modal:

```bash
pip install modal
modal setup
```

If model download requires authentication, create a Modal secret named `huggingface` with your Hugging Face token.

## Running Without Keeping The CLI Open
Use `modal run --detach ...` to submit jobs and return immediately:

```bash
modal run --detach test.py::main_pagekv_expected_attention_static --version 3
```

This is the recommended way to submit longer inference or evaluation jobs.

## Versioned Runs
All Modal artifacts are now versioned.

Artifacts are written under:

```text
/models/runs/<run_tag>/predictions/...
/models/runs/<run_tag>/results/...
/models/runs/<run_tag>/validations/...
```

Run tags are generated as:

```text
v<version>_<timestamp>
```

Example:

```bash
modal run --detach test.py::main_pagekv_expected_attention_static --version 3
```

This prints something like:

```text
Run tag: v3_20260426_153012
```

Evaluate that exact run with:

```bash
modal run --detach test.py::main_eval_pagekv_expected_attention_static --run-tag v3_20260426_153012
modal run --detach test.py::main_csv --run-tag v3_20260426_153012
```

You can also set the tag directly:

```bash
modal run --detach test.py::main_tokenkv_h2o_static --run-tag v7_custom
```

## One-Example Validation
There is a lightweight Modal validation flow for sanity checking method wiring before or after running the full benchmark.

It runs one held-out example and saves:
- prediction
- gold answers
- simple checks such as:
  - `ran_successfully`
  - `pred_file_found`
  - `nonempty_pred`
  - `exact_any`
  - `contains_any_answer`

Example:

```bash
modal run --detach test.py::main_validate_all_static --version 4
```

Validation artifacts are saved under:

```text
/models/runs/<run_tag>/validations/
```

## Project Plan

### Environment setup — Esme
Status: `done`
- Create a conda env with Python 3.10, install transformers >= 4.36, PyTorch (CUDA), and flash-attn.
- Clone the repo and install it.
- Download LLaMA-2-7B and Mistral-7B via Hugging Face.
- Prepare LongBench datasets for evaluation.

### Baseline replication — Esme
Status: `done`
- Run the full-precision model and baseline compression methods on LongBench.
- Log accuracy, memory usage, and FLOPs.
- Confirm baseline speedup / memory trends are reproduced.

### Implement Cluster-based KV — Michael
- Implement a `ClusteredKVCache` module.
- Group key vectors using online k-means.
- Quantize centroids to INT8.
- Retrieve top-K clusters for attention.
- Support:
  - static
  - incremental
  - periodic
- Integrate through Hugging Face monkey-patching.

### Experiments
- Evaluate ClusterKV on LongBench.
- Vary:
  - number of clusters
  - top-K values
- Measure:
  - accuracy
  - memory footprint
  - FLOPs per decode step
- Compare against:
  - uncompressed baseline
  - prompt/page/token ranking variants

### Cluster-approach variations — Collab
- Compare static vs. dynamic clustering
- Focus first on:
  - long prefill
  - short output
- Implement a page-level retrieval variant inspired by Quest
- Explore algorithm-based clustering variants

### Eviction-based / importance ranking methods — Collab
- Heavy-Hitters (H2O)
- High-attention prefill (SnapKV-style)
- Reconstruction error (KVZip-style)
- Expected Attention
- Alternative clustering algorithms:
  - k-means++
  - spherical k-means

### Router (stretch)
- Extract lightweight query features
- Route between ClusterKV and eviction methods
- Evaluate end-to-end tradeoffs

## Method Taxonomy

### KV base method
- no cluster
- paged, static cluster
- cluster by unsupervised algorithms

### Eviction / importance method
- prompt based
- accumulation + top-k
- reconstruction loss

## Current Experiment Names

### Baselines
- `baseline`
- `snapkv_static`
- `quest_static`
- `h2o_static`

### Page-level static methods
- `pagekv_quest_bounds_static`
- `pagekv_snapkv_static`
- `pagekv_h2o_static`
- `pagekv_recon_static`
- `pagekv_expected_attention_static`
- `pagekv_random_static`

### Token-level static methods
- `tokenkv_quest_bounds_static`
- `tokenkv_snapkv_static`
- `tokenkv_h2o_static`
- `tokenkv_recon_static`
- `tokenkv_expected_attention_static`
- `tokenkv_random_static`

### Compatibility aliases
Older `clusterkv_*_static` entrypoints are kept as compatibility aliases and currently point to the page-level family.

## Current Ablation Semantics

### `snapkv_static`
- original prompt-compression baseline in this repo
- static prompt-time compression

### `quest_static`
- separate page-level retrieval path inspired by Quest
- static prompt-time approximation

### `pagekv_*_static`
- score prefix pages
- keep top-scoring pages plus recent window

### `tokenkv_*_static`
- score prefix tokens directly
- keep top-scoring tokens plus recent window

### `h2o_static`
- experiment label exists
- dedicated standalone runner is still stale in this workspace
- nearest runnable approximations are:
  - `pagekv_h2o_static`
  - `tokenkv_h2o_static`

## Starting Data

### Starting benchmark table

| Method | gov_report | hotpotqa | lcc | qasper | Avg | Peak GPU | KV Cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full precision | 32.87 | 42.77 | 55.86 | 32.99 | 41.12 | 23.2 GB | 9,429 MB |
| SnapKV | 30.52 | 42.31 | 55.98 | 33.38 | 40.55 | 20.89 GB | 7,072 MB |
| H2O | 27.28 | 40.38 | 55.90 | 30.97 | 38.63 | 19.53 GB | 5,672 MB |
| ACK (target) | >= 30.5 | >= 42.3 | >= 55.9 | >= 33 | >= 40.5 | < 20.89 GB | < 7,072 MB |

### Quest result snapshot

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| quest | 31.18 | 41.39 | 55.63 | 33.11 | 40.33 |  |  |

## Deliverables
- GitHub repository containing ClusterKV and experiment scripts
- CSV logs of metrics
- plots of trade-offs
- final report
- slide deck or poster

## Open Questions
- theoretical pros / cons
- dynamic clustering practicality
- routing heuristics vs maintenance complexity
