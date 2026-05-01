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
| Michael | `3` | heuristic routing | `v3_20260501_013635` | [ap-rbfehVDnVXvKYjy8Ut08V4](https://modal.com/apps/huxinyu1997/main/ap-rbfehVDnVXvKYjy8Ut08V4) |

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

## Routed Heuristic Experiment
`heuristic_routing` is a single experiment label that routes each example to a backend using only fixed-cost prompt-window keyword checks plus the configured `max_gen`.

It does not use the dataset name to choose the backend. Dataset labels are used only by the LongBench runner to select which split to evaluate.

Current label-free route:
- If `max_gen >= 256` or summary markers appear, use `tokenkv_quest_bounds_dynamic100`.
- Else if sampled prompt windows look code-like, use `clusterattn_recon_static`.
- Else if sampled prompt windows look QA-like and `max_gen >= 128`, use `clusterattn_quest_bounds_static`.
- Else if sampled prompt windows look QA-like, use `pagekv_quest_bounds_static`.
- Else use `clusterattn_recon_static`.

The prompt check samples the first and last prompt windows, so the routing overhead is fixed-cost rather than an `O(n)` scan over the full context. Inference logs print the first few per-example route decisions and a final route-count summary.

Run and evaluate it across all configured datasets:

```bash
modal run --detach test.py::main_heuristic_routing --version 3
modal run --detach test.py::main_eval_heuristic_routing --run-tag v3_20260501_013635
modal run --detach test.py::main_verify_eval_heuristic_routing --run-tag v3_20260501_013635
modal run --detach test.py::main_csv --run-tag v3_20260501_013635
```

One-example validation:

```bash
modal run --detach test.py::main_validate_heuristic_routing --version 1
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
- `tokenkv_quest_bounds_dynamic100`
- `tokenkv_h2o_dynamic`
- `tokenkv_expected_attention_dynamic`
- `tokenkv_random_dynamic`

### Routed heuristic methods
- `heuristic_routing`

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

### Dynamic benchmark snapshot

Current dynamic takeaways:
- Dynamic does not help uniformly.
- On the two tasks where dynamic was expected to matter most, `gov_report` and `qasper`, only `tokenkv_quest_bounds_dynamic` clearly helps relative to its static counterpart:
  - `gov_report`: `29.91` vs `29.76`
  - `qasper`: `33.22` vs `32.97`
- `tokenkv_h2o_dynamic` is close to static on `qasper` (`32.26` vs `32.27`) but still loses on `gov_report` (`28.62` vs `30.03`).
- `pagekv_*_dynamic` and `clusterattn_*_dynamic` regress strongly on `gov_report` and `qasper`, especially the `h2o` variants.
- On `hotpotqa` and `lcc`, dynamic usually does not help either; the best behavior is near-parity from `tokenkv_quest_bounds_dynamic` and `tokenkv_h2o_dynamic`.
- The current dynamic implementation is also much more memory-hungry than the static one:
  - static compressed methods were around `21.87 GB` peak GPU and `8071.6 MB` KV cache
  - dynamic runs are around `33.4 GB` peak GPU and `19.9 GB` KV cache for most methods
- `expected_attention_dynamic` rows are still incomplete in the current CSV and should not be interpreted yet.

Task-by-task comparison against static counterparts:
- `gov_report`
  - `tokenkv_quest_bounds_dynamic`: `+0.15`
  - `tokenkv_h2o_dynamic`: `-1.41`
  - `pagekv_quest_bounds_dynamic`: `-7.55`
  - `pagekv_h2o_dynamic`: `-13.63`
  - `clusterattn_h2o_dynamic`: `-16.38`
- `qasper`
  - `tokenkv_quest_bounds_dynamic`: `+0.25`
  - `tokenkv_h2o_dynamic`: `-0.01`
  - `pagekv_quest_bounds_dynamic`: `-1.54`
  - `pagekv_h2o_dynamic`: `-2.59`
  - `clusterattn_quest_bounds_dynamic`: `-9.83`
  - `clusterattn_h2o_dynamic`: `-14.58`
- `hotpotqa`
  - `tokenkv_quest_bounds_dynamic`: `0.00`
  - `tokenkv_h2o_dynamic`: `-0.25`
  - `pagekv_quest_bounds_dynamic`: `-1.04`
  - `clusterattn_quest_bounds_dynamic`: `-0.96`
- `lcc`
  - `tokenkv_quest_bounds_dynamic`: `-0.23`
  - `tokenkv_h2o_dynamic`: `-0.30`
  - `pagekv_quest_bounds_dynamic`: `-0.19`
  - `clusterattn_quest_bounds_dynamic`: `-1.76`

Interpretation:
- If dynamic is kept at all, `tokenkv_quest_bounds_dynamic` is the strongest current candidate.
- `tokenkv_h2o_dynamic` is the next most defensible dynamic baseline, but it does not beat its static version.
- The page-level and ClusterAttn dynamic variants are currently not competitive on the long summarization / retrieval-heavy tasks that motivated dynamic updates.
- The current `Throughput (tok/s)` column should be treated carefully because it is computed from generated tokens divided by total end-to-end latency, not pure decode-only throughput.

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Avg Prefill Latency (s) | Avg Decode Latency (s) | Max Prefill Latency (s) | Max Decode Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clusterattn_quest_bounds_dynamic |  | 41.21 | 53.82 | 23.7 | 39.58 | 33.41 | 19883.6 | 10.424 | 0.996 | 9.427 | 2.159 | 183.146 | 2.59 | 125.675054 | 1.055184 |
| clusterattn_h2o_dynamic | 13.6 | 35.15 | 52.78 | 17.69 | 29.8 | 33.41 | 19883.9 | 15.704 | 1.006 | 14.698 | 2.528 | 120.25 | 3.03 | 125.705012 | 1.344334 |
| clusterattn_expected_attention_dynamic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| clusterattn_random_dynamic |  | 24.84 | 54.31 | 30.75 | 36.63 | 33.4 | 19880.7 | 7.021 | 0.979 | 6.042 | 2.407 | 140.844 | 4.11 | 125.675063 | 1.561787 |
| pagekv_quest_bounds_dynamic | 23.63 | 40.35 | 55.44 | 31.57 | 37.75 | 33.4 | 19876.2 | 2.628 | 0.857 | 1.772 | 2.038 | 30.852 | 8.63 | 125.675068 | 7.390223 |
| pagekv_h2o_dynamic | 16.29 | 38.81 | 54.63 | 29.88 | 34.9 | 30.98 | 17403.6 | 2.664 | 1.228 | 1.436 | 1.91 | 17.598 | 5.77 | 187.252758 | 11.94751 |
| pagekv_expected_attention_dynamic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| pagekv_random_dynamic | 14.22 | 38.46 | 53.63 | 27.39 | 33.42 | 33.4 | 19879.4 | 2.362 | 0.84 | 1.522 | 1.906 | 23.451 | 10.53 | 125.675063 | 7.759915 |
| tokenkv_quest_bounds_dynamic | 29.91 | 41.91 | 55.59 | 33.22 | 40.16 | 33.4 | 19881.4 | 2.403 | 0.828 | 1.575 | 1.907 | 28.442 | 8.96 | 125.697974 | 7.513029 |
| tokenkv_h2o_dynamic | 28.62 | 39.83 | 55.56 | 32.26 | 39.07 | 33.4 | 19881.5 | 2.524 | 0.846 | 1.678 | 1.924 | 29.005 | 9.12 | 125.84253 | 7.723757 |
| tokenkv_expected_attention_dynamic |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| tokenkv_random_dynamic | 20.83 | 24.84 | 54.31 | 30.75 | 32.68 | 33.4 | 19880.7 | 2.663 | 0.829 | 1.834 | 1.819 | 25.454 | 10.84 | 125.675063 | 8.507744 |

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
