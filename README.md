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

## Full Benchmark Flow
This is the end-to-end workflow for running the full static experiment matrix, then evaluating and summarizing it.
These top-level commands submit one remote orchestrator each, so `--detach` is safe to use for the full batch flow.

1. Submit all inference jobs:

```bash
cd /home/uhdfhnn/Desktop/MLSysProject
modal run --detach test.py::main --version 1
```

2. Note the printed run tag, for example:

```text
v1_20260427_103000
```

3. After inference finishes, submit evaluation for that same run:

```bash
modal run --detach test.py::main_eval_all --run-tag v1_20260427_103000
```

4. Verify the run is complete:

```bash
modal run --detach test.py::main_verify_eval --run-tag v1_20260427_103000
```

5. Generate the CSV summary:

```bash
modal run --detach test.py::main_csv --run-tag v1_20260427_103000
```

What each command does:
- `main`: submits all static-method inference jobs across all datasets
- `main_eval_all`: scores all method/dataset predictions for that run
- `main_verify_eval`: checks that eval files exist and required metrics were logged
- `main_csv`: writes the summary CSV for that run

For a fast sanity check before the full benchmark:

```bash
modal run --detach test.py::main_validate_all_static --version 1
```

Important:
- `main`, `main_eval_all`, and `main_validate_all_static` cover the default full static matrix only.
- Dynamic methods are available, but they are not part of the default full-batch workflow.

## Running Without Keeping The CLI Open
Use `modal run --detach ...` to submit jobs and return immediately:

```bash
modal run --detach test.py::main_pagekv_expected_attention_static --version 3
```

This is the recommended way to submit longer inference or evaluation jobs.

## Run
| Owner | Version | Scope | Run Tag | Link |
| --- | --- | --- | --- | --- |
| Michael | `1` | all static methods | `v1_20260427_030527` | [ap-7D5pMzxNUjZkXQWrnUoZpS](https://modal.com/apps/huxinyu1997/main/ap-7D5pMzxNUjZkXQWrnUoZpS) |

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

To evaluate the full static method matrix for a run:

```bash
modal run --detach test.py::main_eval_all --run-tag v3_20260426_153012
modal run --detach test.py::main_verify_eval --run-tag v3_20260426_153012
modal run --detach test.py::main_csv --run-tag v3_20260426_153012
```

`main_verify_eval` prints `EVAL PASSED` once every expected eval artifact exists for the run.

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

Cluster-specific examples:

```bash
modal run --detach test.py::main_validate_clusterattn_static --version 4
modal run --detach test.py::main_validate_clusterattn_h2o_static --version 4
modal run --detach test.py::main_validate_clusterkv_quest_bounds_static --version 4
modal run --detach test.py::main_validate_clusterkv_quest_bounds_kmeans_static --version 4
modal run --detach test.py::main_validate_clusterkv_expected_attention_spherical_static --version 4
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
- `clusterattn_static`
- `h2o_static`

### ClusterAttn static methods
- `clusterattn_static` (alias to the SnapKV-scored ClusterAttn preset)
- `clusterattn_quest_bounds_static`
- `clusterattn_snapkv_static`
- `clusterattn_h2o_static`
- `clusterattn_recon_static`
- `clusterattn_expected_attention_static`
- `clusterattn_random_static`

### Cluster-level static methods
- `clusterkv_quest_bounds_static` (`kmeanspp` default)
- `clusterkv_quest_bounds_kmeans_static`
- `clusterkv_quest_bounds_spherical_static`
- `clusterkv_snapkv_static`
- `clusterkv_snapkv_kmeans_static`
- `clusterkv_snapkv_spherical_static`
- `clusterkv_h2o_static`
- `clusterkv_h2o_kmeans_static`
- `clusterkv_h2o_spherical_static`
- `clusterkv_recon_static`
- `clusterkv_recon_kmeans_static`
- `clusterkv_recon_spherical_static`
- `clusterkv_expected_attention_static`
- `clusterkv_expected_attention_kmeans_static`
- `clusterkv_expected_attention_spherical_static`
- `clusterkv_random_static`
- `clusterkv_random_kmeans_static`
- `clusterkv_random_spherical_static`

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

### Dynamic methods currently implemented

ClusterAttn dynamic:
- `clusterattn_quest_bounds_dynamic`
- `clusterattn_h2o_dynamic`
- `clusterattn_expected_attention_dynamic`
- `clusterattn_random_dynamic`

PageKV dynamic:
- `pagekv_quest_bounds_dynamic`
- `pagekv_h2o_dynamic`
- `pagekv_expected_attention_dynamic`
- `pagekv_random_dynamic`

TokenKV dynamic:
- `tokenkv_quest_bounds_dynamic`
- `tokenkv_h2o_dynamic`
- `tokenkv_expected_attention_dynamic`
- `tokenkv_random_dynamic`

### Dynamic coverage summary
- Dynamic variants exist for:
  - `clusterattn`
  - `pagekv`
  - `tokenkv`
- Dynamic variants do not exist for:
  - `clusterkv`
  - `snapkv_static`
  - `quest_static`
  - standalone `h2o_static`
- SnapKV-style dynamic hybrids were intentionally removed.
- Reconstruction-error dynamic variants were intentionally removed.
- Expected Attention has both static and dynamic variants.

### Clustering backends
Current real `clusterkv` clustering backends:
- `kmeans`
- `kmeanspp`
- `spherical_kmeans`

The naming convention is:
- no suffix change, e.g. `clusterkv_h2o_static`: `kmeanspp`
- `_kmeans_static`: plain Euclidean k-means
- `_spherical_static`: spherical k-means

## Current Ablation Semantics

### `snapkv_static`
- original prompt-compression baseline in this repo
- static prompt-time compression

### `quest_static`
- separate page-level retrieval path inspired by Quest
- static prompt-time approximation

### `clusterattn_static`
- separate static block-density retrieval path
- uses prompt attention to score tokens, max-pools into blocks, thresholds block centers, expands neighborhoods, then keeps top tokens
- not semantic k-means clustering; this is a ClusterAttention-style baseline

### `clusterattn_*_static`
- same ClusterAttn block-density retrieval pipeline
- swap only the token scoring backend:
  - `quest_bounds`
  - `snapkv_prefill`
  - `h2o_accum`
  - `reconstruction_error`
  - `expected_attention`
  - `random`

### `clusterkv_*_static`
- real static cluster selection path
- cluster prefix keys per `(batch, head)` using a k-means-family backend
- score tokens with the selected ranking backend
- aggregate token scores to cluster scores
- recall top clusters until the token budget is filled

Examples:
- `clusterkv_quest_bounds_static`: `quest_bounds` ranking + `kmeanspp`
- `clusterkv_quest_bounds_kmeans_static`: `quest_bounds` ranking + `kmeans`
- `clusterkv_quest_bounds_spherical_static`: `quest_bounds` ranking + `spherical_kmeans`
- `clusterkv_h2o_static`: `h2o_accum` ranking + `kmeanspp`
- `clusterkv_expected_attention_spherical_static`: `expected_attention` ranking + `spherical_kmeans`

### `pagekv_*_static`
- score prefix pages
- keep top-scoring pages plus recent window

### `tokenkv_*_static`
- score prefix tokens directly
- keep top-scoring tokens plus recent window

### `clusterattn_*_dynamic`, `pagekv_*_dynamic`, `tokenkv_*_dynamic`
- dynamic decode-time eviction variants for the non-`clusterkv` families
- they recompress the retained prefix during generation instead of compressing only once at prefill
- `h2o` and `quest_bounds` are the most natural dynamic variants
- `expected_attention_dynamic` uses a periodic refresh design with buffered recent queries

### `h2o_static`
- experiment label exists
- dedicated standalone runner is still stale in this workspace
- nearest runnable approximations are:
  - `clusterkv_h2o_static`
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

### Current benchmark snapshot

Best current compressed method:
- `clusterattn_recon_static`
- Average score: `40.56` vs baseline `41.12`
- Peak GPU: `21.87 GB` vs baseline `25.93 GB` for a `15.7%` reduction
- KV cache: `8,071.6 MB` vs baseline `12,223.1 MB` for a `34.0%` reduction
- Average latency: `5.374 s` vs baseline `7.733 s` for a `30.5%` reduction
- Profiled TFLOPs: `169.579216` vs baseline `169.639661` with no meaningful reduction

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.87 | 42.77 | 55.86 | 32.99 | 41.12 | 25.93 | 12223.1 | 7.733 | 15.6 | 169.639661 | 7.041622 |
| snapkv_static | 30.52 | 42.31 | 55.98 | 33.38 | 40.55 | 21.87 | 8071.6 | 6.017 | 14.6 | 169.902572 | 4.928252 |
| quest_static | 31.18 | 41.39 | 55.63 | 33.11 | 40.33 | 21.87 | 8071.6 | 5.396 | 17.25 | 170.151672 | 7.350395 |
| clusterattn_static | 30.3 | 42.69 | 55.82 | 33.19 | 40.5 | 21.87 | 8071.6 | 7.856 | 13.9 | 170.492271 | 4.64913 |
| clusterattn_quest_bounds_static | 30.52 | 42.17 | 55.58 | 33.53 | 40.45 | 21.87 | 8071.6 | 5.552 | 17.3 | 170.336097 | 4.833747 |
| clusterattn_snapkv_static | 30.3 | 42.69 | 55.82 | 33.19 | 40.5 | 21.87 | 8071.6 | 7.834 | 13.38 | 170.492271 | 4.747719 |
| clusterattn_h2o_static | 29.98 | 40.33 | 55.86 | 32.27 | 39.61 | 21.87 | 8071.6 | 5.569 | 17.09 | 169.74129 | 5.114582 |
| clusterattn_recon_static | 30.48 | 42.1 | 56.21 | 33.47 | 40.56 | 21.87 | 8071.6 | 5.374 | 18.24 | 169.579216 | 5.083752 |
| clusterattn_expected_attention_static | 30.16 | 42.12 | 55.78 | 32.48 | 40.13 | 21.87 | 8071.6 | 5.298 | 16.75 | 169.827199 | 5.472103 |
| clusterattn_random_static | 22.13 | 23.45 | 54.88 | 28.9 | 32.34 | 21.87 | 8071.6 | 8.309 | 15.88 | 170.624572 | 5.295625 |
| pagekv_quest_bounds_static | 31.18 | 41.39 | 55.63 | 33.11 | 40.33 | 21.87 | 8071.6 | 5.574 | 16.84 | 170.151672 | 6.64543 |
| pagekv_snapkv_static | 30.91 | 42.4 | 55.68 | 32.93 | 40.48 | 21.87 | 8071.6 | 6.915 | 14.42 | 170.481604 | 7.693576 |
| pagekv_h2o_static | 29.92 | 42.64 | 55.63 | 32.47 | 40.16 | 21.87 | 8071.6 | 4.974 | 18.73 | 170.49864 | 6.861369 |
| pagekv_recon_static | 29.45 | 37.78 | 56.01 | 30.24 | 38.37 | 21.87 | 8071.6 | 8.015 | 14.98 | 170.56057 | 6.746171 |
| pagekv_expected_attention_static | 29.45 | 42.32 | 55.56 | 32.05 | 39.84 | 21.87 | 8071.6 | 4.864 | 17.26 | 169.674306 | 6.91476 |
| pagekv_random_static | 24.02 | 28.36 | 54.52 | 31.14 | 34.51 | 21.87 | 8071.6 | 7.556 | 17.37 | 170.631683 | 7.261989 |
| tokenkv_quest_bounds_static | 29.76 | 41.91 | 55.82 | 32.97 | 40.12 | 21.87 | 8071.6 | 6.873 | 14.34 | 170.318319 | 6.544338 |
| tokenkv_snapkv_static | 30.34 | 42.52 | 55.79 | 33.01 | 40.41 | 21.87 | 8071.6 | 6.654 | 17.07 | 170.531383 | 9.015235 |
| tokenkv_h2o_static | 30.03 | 40.08 | 55.86 | 32.27 | 39.56 | 21.87 | 8071.6 | 4.818 | 19.54 | 169.74129 | 9.276461 |
| tokenkv_recon_static | 30.48 | 42.1 | 56.21 | 33.47 | 40.56 | 21.87 | 8071.6 | 5.396 | 17.56 | 169.579216 | 7.467223 |
| tokenkv_expected_attention_static | 30.51 | 41.95 | 55.91 | 32.98 | 40.34 | 21.87 | 8071.6 | 5.087 | 18.6 | 170.186318 | 7.548452 |
| tokenkv_random_static | 22.13 | 23.45 | 54.88 | 28.9 | 32.34 | 21.87 | 8071.6 | 6.889 | 16.75 | 170.624572 | 6.89309 |

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
