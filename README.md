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
| Michael | `4` | heuristic routing | `v4_20260502_164454` | [ap-QZ1Wk2M84eKGIJntqjenCX](https://modal.com/apps/huxinyu1997/main/ap-QZ1Wk2M84eKGIJntqjenCX) |
| Michael | `5` | XGBoost router data, 100 examples per dataset | `v5_20260502_164834` | [ap-eaVZCLvPr7rREX6JZvq2w3](https://modal.com/apps/huxinyu1997/main/ap-eaVZCLvPr7rREX6JZvq2w3) |
| Michael | `6` | heuristic routing, fixed cache state and generate-only memory | `v6_20260502_173239` | [ap-PkBinoedjo9WMYx0RvK1JR](https://modal.com/apps/huxinyu1997/main/ap-PkBinoedjo9WMYx0RvK1JR) |
| Michael | `7` | heuristic routing, post diagnostic-prefill memory fix | `v7_20260503_032528` | [ap-b5AnUvWbuvi5kDyPiGBQR4](https://modal.com/apps/huxinyu1997/main/ap-b5AnUvWbuvi5kDyPiGBQR4) |
| Michael | `8` | XGBoost routing using router trained from `v5_20260502_164834` | `v8_20260503_033648` | [ap-wMCiJ7RmER9aBxmseeeJWU](https://modal.com/apps/huxinyu1997/main/ap-wMCiJ7RmER9aBxmseeeJWU) |
| Michael | `9` | refreshed static methods, no `clusterkv_*`, latest metrics | `v9_20260503_152041` | [ap-Rin9rd1bM7WnEPRffR5acR](https://modal.com/apps/huxinyu1997/main/ap-Rin9rd1bM7WnEPRffR5acR) |
| Michael | `12` | baseline cluster-path control on A100-80GB | `v12_20260504_042828` | pending |

### Refreshed Static Result: `v9_20260503_152041`

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Avg Prefill Latency (s) | Avg Decode Latency (s) | Max Prefill Latency (s) | Max Decode Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.87 | 42.77 | 55.86 | 32.99 | 41.12 | 56.21 | 43232.3 | 6.209 | 0.936 | 5.273 | 3.234 | 61.030 | 16.91 | 169.639661 | 7.255866 |
| snapkv_static | 30.52 | 42.31 | 55.98 | 33.38 | 40.55 | 21.87 | 8071.6 | 6.084 | 0.879 | 5.205 | 3.012 | 89.932 | 15.55 | 169.902572 | 6.899190 |
| quest_static | 31.18 | 41.39 | 55.63 | 33.11 | 40.33 | 21.87 | 8071.6 | 6.177 | 0.888 | 5.289 | 3.057 | 111.505 | 15.20 | 170.151672 | 6.470891 |
| clusterattn_static | 30.30 | 42.69 | 55.82 | 33.19 | 40.50 | 31.53 | 17963.7 | 6.377 | 1.012 | 5.364 | 3.272 | 129.804 | 14.75 | 170.492271 | 5.259820 |
| clusterattn_quest_bounds_static | 30.52 | 42.17 | 55.58 | 33.53 | 40.45 | 31.53 | 17963.7 | 7.123 | 1.067 | 6.056 | 3.331 | 148.881 | 14.00 | 170.336097 | 5.080633 |
| clusterattn_snapkv_static | 30.30 | 42.69 | 55.82 | 33.19 | 40.50 | 31.53 | 17963.7 | 7.222 | 1.008 | 6.214 | 3.276 | 151.226 | 13.79 | 170.492271 | 5.088779 |
| clusterattn_h2o_static | 29.98 | 40.33 | 55.86 | 32.27 | 39.61 | 31.53 | 17963.2 | 6.395 | 0.991 | 5.404 | 3.270 | 78.564 | 15.54 | 169.741290 | 6.079391 |
| clusterattn_recon_static | 30.48 | 42.10 | 56.21 | 33.47 | 40.56 | 31.53 | 17962.9 | 5.863 | 0.994 | 4.869 | 3.302 | 72.350 | 16.59 | 169.579216 | 5.969799 |
| clusterattn_expected_attention_static | 30.24 | 42.24 | 55.77 | 32.54 | 40.20 | 31.53 | 17963.2 | 5.926 | 1.001 | 4.925 | 3.285 | 111.118 | 14.76 | 170.097451 | 4.617068 |
| clusterattn_random_static | 21.93 | 26.24 | 54.31 | 30.81 | 33.32 | 31.53 | 17963.2 | 8.474 | 0.973 | 7.501 | 3.239 | 169.351 | 15.13 | 170.624572 | 5.886126 |
| pagekv_quest_bounds_static | 31.18 | 41.39 | 55.63 | 33.11 | 40.33 | 31.53 | 17962.9 | 6.641 | 0.887 | 5.754 | 3.139 | 118.087 | 16.43 | 170.151672 | 7.812696 |
| pagekv_snapkv_static | 30.91 | 42.40 | 55.68 | 32.93 | 40.48 | 31.53 | 17962.9 | 7.133 | 0.884 | 6.249 | 3.145 | 147.268 | 14.86 | 170.481604 | 6.921557 |
| pagekv_h2o_static | 29.92 | 42.64 | 55.63 | 32.47 | 40.16 | 31.53 | 17962.9 | 6.410 | 0.903 | 5.507 | 3.171 | 139.668 | 15.13 | 170.498640 | 7.545716 |
| pagekv_recon_static | 29.45 | 37.78 | 56.01 | 30.24 | 38.37 | 31.53 | 17962.9 | 6.428 | 0.891 | 5.538 | 3.177 | 128.231 | 17.16 | 170.560570 | 7.981664 |
| pagekv_expected_attention_static | 30.37 | 42.75 | 55.83 | 31.19 | 40.03 | 31.53 | 17962.9 | 5.475 | 0.900 | 4.576 | 3.181 | 81.564 | 16.25 | 169.788212 | 7.170652 |
| pagekv_random_static | 23.44 | 29.52 | 54.74 | 31.93 | 34.91 | 31.53 | 17962.9 | 6.616 | 0.881 | 5.735 | 3.129 | 134.320 | 16.35 | 170.642350 | 7.537026 |
| tokenkv_quest_bounds_static | 29.76 | 41.91 | 55.82 | 32.97 | 40.12 | 31.53 | 17963.7 | 5.752 | 0.868 | 4.883 | 3.087 | 114.879 | 16.29 | 170.318319 | 7.091499 |
| tokenkv_snapkv_static | 30.34 | 42.52 | 55.79 | 33.01 | 40.41 | 31.53 | 17963.7 | 4.973 | 0.901 | 4.071 | 3.162 | 105.885 | 18.14 | 170.531383 | 6.992071 |
| tokenkv_h2o_static | 30.03 | 40.08 | 55.86 | 32.27 | 39.56 | 31.53 | 17963.7 | 6.341 | 0.883 | 5.458 | 3.119 | 76.120 | 15.97 | 169.741290 | 7.630256 |
| tokenkv_recon_static | 30.48 | 42.10 | 56.21 | 33.47 | 40.56 | 31.53 | 17962.9 | 6.787 | 0.877 | 5.910 | 3.114 | 78.809 | 14.95 | 169.579216 | 7.233228 |
| tokenkv_expected_attention_static | 30.80 | 41.33 | 55.97 | 32.82 | 40.23 | 31.53 | 17963.2 | 5.849 | 0.880 | 4.969 | 3.071 | 122.082 | 15.49 | 170.453014 | 7.047233 |
| tokenkv_random_static | 21.93 | 26.24 | 54.31 | 30.81 | 33.32 | 31.53 | 17963.7 | 7.689 | 0.872 | 6.818 | 3.086 | 147.864 | 17.53 | 170.624572 | 7.663775 |

Full CSV: `/models/runs/v9_20260503_152041/results/summary.csv`.

Analysis:
- Accuracy is stable relative to the earlier static run. The best average remains `40.56`, reached by `clusterattn_recon_static` and `tokenkv_recon_static`.
- `baseline` now denotes the full-KV cluster-path control from `v12_20260504_042828`, so systems metrics are comparable to the generalized `clusterattn`/`pagekv`/`tokenkv` path. Accuracy is unchanged from native full precision at `41.12`.
- The generalized `clusterattn`/`pagekv`/`tokenkv` backend now reports about `31.53 GB` peak and `~17.96 GB` extra generate-path memory. This should be interpreted as extra generate-path peak memory, not pure retained KV size.
- `tokenkv_snapkv_static` is the fastest useful compressed static method in this refresh (`4.973s`, `18.14 tok/s`) while keeping average accuracy at `40.41`.
- Total profiled TFLOPs is effectively unchanged across static methods, so systems differences are mostly visible in memory, latency, throughput, and TFLOPs/s.

### Full-KV Cluster-Path Baseline: `v12_20260504_042828`

This run uses `baseline_clusterpath_static` on A100-80GB and serves as the systems baseline for generalized-cache comparisons. It keeps full KV while entering the same `clusterkv` monkeypatch path used by the generalized methods.

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Avg Prefill Latency (s) | Avg Decode Latency (s) | Max Prefill Latency (s) | Max Decode Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_clusterpath_static | 32.87 | 42.77 | 55.86 | 32.99 | 41.12 | 56.21 | 43232.3 | 6.209 | 0.936 | 5.273 | 3.234 | 61.030 | 16.91 | 169.639661 | 7.255866 |

For analysis plots and merged tables, this row replaces the native HuggingFace baseline row so the baseline uses the same generalized cache path as the proposed methods.

### Merged Static Result: `v1_20260427_030527` + `v9_20260503_152041`

For non-baseline rows, columns present in both static runs report the arithmetic mean. For columns missing from the earlier run, the table uses the `v9_20260503_152041` value. The baseline row uses the full-KV cluster-path baseline from `v12_20260504_042828` for a same-path systems comparison. The same data is available to `analysis.py` at `figures/static_merged_v1_v9.csv`; when `analysis.py` is run without `--static`, it now reads this merged README table by default.

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Avg Prefill Latency (s) | Avg Decode Latency (s) | Max Prefill Latency (s) | Max Decode Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.87 | 42.77 | 55.86 | 32.99 | 41.12 | 56.21 | 43232.3 | 6.209 | 0.936 | 5.273 | 3.234 | 61.030 | 16.91 | 169.639661 | 7.255866 |
| snapkv_static | 30.52 | 42.31 | 55.98 | 33.38 | 40.55 | 21.87 | 8071.6 | 6.050 | 0.879 | 5.205 | 3.012 | 89.932 | 15.07 | 169.902572 | 5.913721 |
| quest_static | 31.18 | 41.39 | 55.63 | 33.11 | 40.33 | 21.87 | 8071.6 | 5.787 | 0.888 | 5.289 | 3.057 | 111.505 | 16.23 | 170.151672 | 6.910643 |
| clusterattn_static | 30.30 | 42.69 | 55.82 | 33.19 | 40.50 | 26.70 | 13017.7 | 7.117 | 1.012 | 5.364 | 3.272 | 129.804 | 14.32 | 170.492271 | 4.954475 |
| clusterattn_quest_bounds_static | 30.52 | 42.17 | 55.58 | 33.53 | 40.45 | 26.70 | 13017.7 | 6.338 | 1.067 | 6.056 | 3.331 | 148.881 | 15.65 | 170.336097 | 4.957190 |
| clusterattn_snapkv_static | 30.30 | 42.69 | 55.82 | 33.19 | 40.50 | 26.70 | 13017.7 | 7.528 | 1.008 | 6.214 | 3.276 | 151.226 | 13.59 | 170.492271 | 4.918249 |
| clusterattn_h2o_static | 29.98 | 40.33 | 55.86 | 32.27 | 39.61 | 26.70 | 13017.4 | 5.982 | 0.991 | 5.404 | 3.270 | 78.564 | 16.31 | 169.741290 | 5.596986 |
| clusterattn_recon_static | 30.48 | 42.10 | 56.21 | 33.47 | 40.56 | 26.70 | 13017.2 | 5.619 | 0.994 | 4.869 | 3.302 | 72.350 | 17.41 | 169.579216 | 5.526775 |
| clusterattn_expected_attention_static | 30.20 | 42.18 | 55.78 | 32.51 | 40.17 | 26.70 | 13017.4 | 5.612 | 1.001 | 4.925 | 3.285 | 111.118 | 15.75 | 169.962325 | 5.044586 |
| clusterattn_random_static | 22.03 | 24.84 | 54.59 | 29.85 | 32.83 | 26.70 | 13017.4 | 8.392 | 0.973 | 7.501 | 3.239 | 169.351 | 15.51 | 170.624572 | 5.590876 |
| pagekv_quest_bounds_static | 31.18 | 41.39 | 55.63 | 33.11 | 40.33 | 26.70 | 13017.2 | 6.107 | 0.887 | 5.754 | 3.139 | 118.087 | 16.63 | 170.151672 | 7.229063 |
| pagekv_snapkv_static | 30.91 | 42.40 | 55.68 | 32.93 | 40.48 | 26.70 | 13017.2 | 7.024 | 0.884 | 6.249 | 3.145 | 147.268 | 14.64 | 170.481604 | 7.307567 |
| pagekv_h2o_static | 29.92 | 42.64 | 55.63 | 32.47 | 40.16 | 26.70 | 13017.2 | 5.692 | 0.903 | 5.507 | 3.171 | 139.668 | 16.93 | 170.498640 | 7.203542 |
| pagekv_recon_static | 29.45 | 37.78 | 56.01 | 30.24 | 38.37 | 26.70 | 13017.2 | 7.222 | 0.891 | 5.538 | 3.177 | 128.231 | 16.07 | 170.560570 | 7.363918 |
| pagekv_expected_attention_static | 29.91 | 42.53 | 55.70 | 31.62 | 39.94 | 26.70 | 13017.2 | 5.169 | 0.900 | 4.576 | 3.181 | 81.564 | 16.76 | 169.731259 | 7.042706 |
| pagekv_random_static | 23.73 | 28.94 | 54.63 | 31.54 | 34.71 | 26.70 | 13017.2 | 7.086 | 0.881 | 5.735 | 3.129 | 134.320 | 16.86 | 170.637017 | 7.399508 |
| tokenkv_quest_bounds_static | 29.76 | 41.91 | 55.82 | 32.97 | 40.12 | 26.70 | 13017.7 | 6.312 | 0.868 | 4.883 | 3.087 | 114.879 | 15.31 | 170.318319 | 6.817918 |
| tokenkv_snapkv_static | 30.34 | 42.52 | 55.79 | 33.01 | 40.41 | 26.70 | 13017.7 | 5.813 | 0.901 | 4.071 | 3.162 | 105.885 | 17.61 | 170.531383 | 8.003653 |
| tokenkv_h2o_static | 30.03 | 40.08 | 55.86 | 32.27 | 39.56 | 26.70 | 13017.7 | 5.579 | 0.883 | 5.458 | 3.119 | 76.120 | 17.75 | 169.741290 | 8.453359 |
| tokenkv_recon_static | 30.48 | 42.10 | 56.21 | 33.47 | 40.56 | 26.70 | 13017.2 | 6.091 | 0.877 | 5.910 | 3.114 | 78.809 | 16.25 | 169.579216 | 7.350226 |
| tokenkv_expected_attention_static | 30.66 | 41.64 | 55.94 | 32.90 | 40.28 | 26.70 | 13017.4 | 5.468 | 0.880 | 4.969 | 3.071 | 122.082 | 17.05 | 170.319666 | 7.297842 |
| tokenkv_random_static | 22.03 | 24.84 | 54.59 | 29.85 | 32.83 | 26.70 | 13017.7 | 7.289 | 0.872 | 6.818 | 3.086 | 147.864 | 17.14 | 170.624572 | 7.278433 |

### Heuristic Routing Result: Latest Verified-Code Run

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Avg Prefill Latency (s) | Avg Decode Latency (s) | Max Prefill Latency (s) | Max Decode Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heuristic_routing | 28.97 | 41.18 | 56.21 | 33.53 | 39.97 | 31.53 | 17963.7 | 7.848 | 0.950 | 6.898 | 3.446 | 182.160 | 14.80 | 170.623406 | 5.701121 |

This row is from `/models/runs/v7_20260503_032528/results/summary.csv` after the diagnostic-prefill output was deleted before generate-path peak-memory measurement. The memory columns are now comparable to the XGBoost routed run, but still represent extra generate-path peak memory rather than pure retained KV size.

Analysis:
- Accuracy is in the expected range: `39.97` average, below the better compressed static methods and below XGBoost routing.
- The router is strong on `lcc` (`56.21`) and `qasper` (`33.53`), but weak on `gov_report` (`28.97`).
- Average latency is `7.848s`, with a large max decode tail (`182.160s`), so this run is no longer a clear latency win after using the corrected measurement path.

### XGBoost Routing Result: `v8_20260503_033648`

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Avg Prefill Latency (s) | Avg Decode Latency (s) | Max Prefill Latency (s) | Max Decode Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgb_routing | 32.40 | 44.06 | 56.06 | 34.44 | 41.74 | 31.53 | 17963.7 | 5.670 | 0.907 | 4.763 | 3.283 | 104.402 | 18.12 | 170.339653 | 7.168930 |

Analysis:
- This is the strongest routed result so far: `41.74` average, above the static full-precision baseline average (`41.12`) and above the heuristic router (`40.00`).
- Gains are broad: `gov_report=32.40`, `hotpotqa=44.06`, and `qasper=34.44` are all better than the heuristic run; `lcc=56.06` remains close to the best static/routed numbers.
- Latency is higher than heuristic routing (`5.67s` vs `2.23s`) but still in the same range as many static compressed methods.
- Memory should be interpreted as the latest generate-path measurement. It is higher than fixed single-method static runs because the routed method can switch backend configs across examples and retains the largest observed route/runtime footprint.

### Heuristic Routing Result: `v3_20260501_013635`

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Avg Prefill Latency (s) | Avg Decode Latency (s) | Max Prefill Latency (s) | Max Decode Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heuristic_routing | 29.91 | 54.55 | 56.29 | 51.14 | 47.97 | 50.7 | 37592.3 | 2.274 | 0.417 | 1.857 | 4.135 | 46.942 | 27.55 | 254.137696 | 6.77236 |

Verification status: incomplete. `main_verify_eval_heuristic_routing` reports missing logged metrics for `qasper`, `hotpotqa`, and `gov_report`, so the score row is recorded for reference but the memory, latency, throughput, and FLOPs columns are not reliable for this run yet.

Analysis:
- Accuracy is much higher than the previous full static benchmark average (`47.97` vs roughly `41`), but `hotpotqa=54.55` and `qasper=51.14` are large jumps and should be treated as suspicious until the run is re-verified against complete inference metric logs.
- Systems metrics appear to be aggregated from incomplete logging. The reported `50.7 GB` peak GPU and `37.6 GB` KV cache should not be compared directly with the earlier static/dynamic tables.
- Before using this as a headline result, rerun or repair heuristic inference for the same run tag and require `EVAL PASSED`.

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

## Baseline Cluster-Path Control
`baseline_clusterpath_static` is a control experiment for measuring the overhead of the generalized `ClusterKVCache`/monkeypatch path without top-k compression. It uses `baseline_clusterpath_fullkv.json`, which sets `max_capacity_prompt=1000000`, so `update_kv()` should keep the full KV for normal LongBench contexts while still entering the `clusterkv` code path.

Run it across all configured datasets on A100-80GB. The 40GB A100 path can OOM on long `lcc` examples because this control intentionally disables compression inside the generalized cache path.

```bash
modal run --detach test.py::main_baseline_clusterpath_static_a100_80gb --version <version>
```

Then evaluate and summarize:

```bash
modal run --detach test.py::main_eval_baseline_clusterpath_static --run-tag <run_tag>
modal run test.py::main_verify_eval_method --method baseline_clusterpath_static --run-tag <run_tag>
modal run test.py::main_csv --run-tag <run_tag>
```

Use this to compare against `baseline` when checking whether high generate-path memory comes from the generalized cache path itself rather than top-k compression.

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

## XGBoost Router Data Collection
The learned-router data collection runs a small candidate method set per example, then joins predictions into per-example training rows for an XGBoost method selector.

Candidate methods:
- `tokenkv_quest_bounds_dynamic100`
- `clusterattn_recon_static`
- `clusterattn_quest_bounds_static`
- `pagekv_quest_bounds_static`
- `tokenkv_quest_bounds_static`
- `tokenkv_h2o_dynamic`

End-to-end flow:

1. Collect candidate predictions on 100 examples per dataset:

```bash
modal run --detach test.py::main_xgb_router_data_100 --version 1
```

This prints a training-data run tag such as `v5_20260502_164834`.

2. Evaluate candidate predictions:

```bash
modal run --detach test.py::main_eval_xgb_router_data --run-tag <run_tag>
```

3. Verify all candidate eval artifacts and metrics are present:

```bash
modal run --detach test.py::main_verify_eval_xgb_router_data --run-tag <run_tag>
```

4. Build per-example supervised router rows:

```bash
modal run --detach test.py::main_build_xgb_router_data --run-tag <run_tag>
```

`main_xgb_router_data_100` runs 100 examples per dataset, so the default collection size is:

```text
6 candidate methods * 4 datasets * 100 examples = 2400 candidate predictions
```

Output:

```text
/models/runs/<run_tag>/results/router_data/xgb_candidates.jsonl
/models/runs/<run_tag>/results/router_data/xgb_candidates_summary.json
```

Each JSONL row contains prompt-window features, per-candidate prediction score, latency fields when available, and the best candidate label. Use `--latency-weight` on `main_build_xgb_router_data` to label by `score - latency_weight * latency_s`; the default labels by score only.

Example using the completed router-data run:

```bash
modal run --detach test.py::main_eval_xgb_router_data --run-tag v5_20260502_164834
modal run --detach test.py::main_verify_eval_xgb_router_data --run-tag v5_20260502_164834
modal run --detach test.py::main_build_xgb_router_data --run-tag v5_20260502_164834
```

### Train and Deploy XGBoost Router
Train the router after `xgb_candidates.jsonl` has been built:

```bash
modal run --detach test.py::main_train_xgb_router --run-tag <router_data_run_tag>
```

For the completed router-data run:

```bash
modal run --detach test.py::main_train_xgb_router --run-tag v5_20260502_164834
```

By default, training uses task-agnostic prompt features only: `max_gen`, prompt length/statistics, keyword-marker counts, and example length. It does not use dataset name unless explicitly enabled:

```bash
modal run --detach test.py::main_train_xgb_router --run-tag <router_data_run_tag> --include-dataset
```

Test-time router features:
- `max_gen`: configured max output tokens for the prompt/task.
- `prompt_chars`: full formatted prompt character count.
- `sample_chars`: character count of the sampled prompt window used for marker detection.
- `code_marker_hits`: count of code markers in the sampled prompt window.
- `summary_marker_hits`: count of summarization markers in the sampled prompt window.
- `qa_marker_hits`: count of QA/passage markers in the sampled prompt window.
- `newline_ratio`: fraction of prompt characters that are newlines.
- `digit_ratio`: fraction of prompt characters that are digits.
- `punct_ratio`: fraction of prompt characters that are punctuation.
- `length`: LongBench-provided input length metadata when present.

Candidate scores, candidate latencies, generated tokens, context length, and prediction text are not test-time router inputs. They are used only during data construction to choose the supervised label, usually `best_utility_method` or `best_score_method`.

The trained artifacts are saved in:

```text
/models/runs/<router_data_run_tag>/results/router_data/xgb_router.json
/models/runs/<router_data_run_tag>/results/router_data/xgb_router_metadata.json
/models/runs/<router_data_run_tag>/results/router_data/xgb_router_metrics.json
```

Validate the deployed router on one example:

```bash
modal run --detach test.py::main_validate_xgb_routing --router-run-tag <router_data_run_tag> --version <version>
```

Example:

```bash
modal run --detach test.py::main_validate_xgb_routing --router-run-tag v5_20260502_164834 --version 1
```

Run full routed inference, then eval and verify:

```bash
modal run --detach test.py::main_xgb_routing --router-run-tag <router_data_run_tag> --version <version>
modal run --detach test.py::main_eval_xgb_routing --run-tag <xgb_routing_run_tag>
modal run --detach test.py::main_verify_eval_xgb_routing --run-tag <xgb_routing_run_tag>
modal run --detach test.py::main_csv --run-tag <xgb_routing_run_tag>
```

Example:

```bash
modal run --detach test.py::main_xgb_routing --router-run-tag v5_20260502_164834 --version 1
modal run --detach test.py::main_eval_xgb_routing --run-tag <xgb_routing_run_tag>
modal run --detach test.py::main_verify_eval_xgb_routing --run-tag <xgb_routing_run_tag>
modal run --detach test.py::main_csv --run-tag <xgb_routing_run_tag>
```

Deployment flow: `main_xgb_routing` loads `/models/runs/<router_data_run_tag>/results/router_data/xgb_router.json`, predicts a candidate method per example, applies that candidate's existing KV config, and saves normal LongBench predictions under the new `<xgb_routing_run_tag>`.

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
- Peak GPU: `21.87 GB` vs baseline `56.21 GB` for a `61.1%` reduction
- KV cache: `8,071.6 MB` vs baseline `43,232.3 MB` for an `81.3%` reduction
- Average latency: `5.374 s` vs baseline `6.209 s` for a `13.4%` reduction
- Profiled TFLOPs: `169.579216` vs baseline `169.639661` with no meaningful reduction

| Method | gov_report | hotpotqa | lcc | qasper | Average | Peak GPU (GB) | KV Cache (MB) | Avg Latency (s) | Throughput (tok/s) | Profiled TFLOPs | Profiled TFLOPs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.87 | 42.77 | 55.86 | 32.99 | 41.12 | 56.21 | 43232.3 | 6.209 | 16.91 | 169.639661 | 7.255866 |
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
