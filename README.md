# SnapKV :camera:
We introduce an innovative and out-of-box KV cache compression method, [SnapKV](https://arxiv.org/abs/2404.14469).
## Requirements
Currently tested with `transformers==4.37.0`, need to check if it is compatible with higher version.
```
transformers>=4.36
flash-attn==2.4.0
```
## Installation
```
git clone git@github.com:FasterDecoding/SnapKV.git
cd SnapKV
pip install -e .
```
## Quick Start
### Use SnapKV-optimized Models
For example: 
```python
from snapkv.monkeypatch.monkeypatch import replace_mistral
replace_mistral() # Use monkey patches enable SnapKV
```

Check [the example notebook](./notebooks/example.ipynb).

### Customize Your SnapKV-optimized Models
SnapKV can be easily integrated with other models. 

You can follow the comment marked with `[SnapKV]` in [existing models](./snapkv/monkeypatch/monkeypatch.py) to construct your own models. (Currently we support [Llama family](./snapkv/monkeypatch/llama_hijack_4_37.py)/ [Mistral](./snapkv/monkeypatch//mistral_hijack_4_37.py)/ [Mixtral](./snapkv/monkeypatch//mixtral_hijack_4_37.py)) 

The detailed algorithm of SnapKV is in [`snapkv_utils.py`](./snapkv/monkeypatch/snapkv_utils.py)


## Partial Results
![Comprehensive Experiment Results on LongBench](./assets/longbench.jpg)
![Pressure Test Result on Needle-in-a-Haystack](./assets/LWM-Text-Chat-1M_SnapKV.jpg)

## Current Ablations In This Repo
The repository now contains several experiment-time KV compression ablations beyond the original SnapKV path. The descriptions below explain how they are currently implemented in this codebase, which is not always identical to the original paper.

### `baseline`
- No KV compression.
- Uses the model with standard FlashAttention-based decoding.
- LongBench runner: `experiments/LongBench/pred_snap.py` without `--compress_args_path`.

### `snapkv_static`
- Original SnapKV-style prompt compression path from this repository.
- Uses an observation window near the end of the prompt to score prompt tokens, then keeps a compressed prefix plus a recent window.
- Configs use:
  - `window_size`
  - `max_capacity_prompt`
  - `kernel_size`
  - `pooling`
- Main implementation: [`snapkv_utils.py`](./snapkv/monkeypatch/snapkv_utils.py)

### `quest_static`
- Separate page-level retrieval path inspired by Quest.
- The current implementation is static prompt-time compression in this repo:
  - split the prefix KV into fixed-size pages,
  - compute per-page key `min/max`,
  - score each page with `sum(max(q * min, q * max))` using the current query,
  - keep top pages from the prefix plus the recent window.
- This is implemented in [`questkv_utils.py`](./snapkv/monkeypatch/questkv_utils.py).
- Important: this is not yet a full decode-time Quest implementation. It is a static prompt-time approximation.

### `pagekv_*_static`
These ablations run through the static ranking scaffold with `selection_granularity = page`.

Shared structure:
- split the prefix into fixed-size pages,
- score pages with one of the ranking backends below,
- keep the top-scoring prefix pages plus the recent window.

Shared config fields:
- `window_size`
- `max_capacity_prompt`
- `page_size`
- `ranking_backend`
- `observation_window`
- `selection_granularity = page`

Current page-level static ranking ablations:

#### `pagekv_quest_bounds_static`
- Uses the same Quest-style page bound score as the `quest` method.
- Per-page score:
  - compute page key `min/max`,
  - score with `sum(max(q * min, q * max))`.
- Main code: [`clusterkv_utils.py`](./snapkv/monkeypatch/clusterkv_utils.py)

#### `pagekv_snapkv_static`
- SnapKV-like prefill attention ranking inside the `clusterkv` experiment path.
- Uses the last `observation_window` prompt queries, computes attention to prefix tokens, averages those prompt-time attention weights, and converts token scores to page scores by page max.
- This is a static approximation of "high-attention prefill".

#### `pagekv_h2o_static`
- H2O-like heavy-hitter ranking inside the `clusterkv` experiment path.
- Uses uniformly sampled prompt queries, computes attention to prefix tokens, accumulates those attention weights across the sampled queries, and converts token scores to page scores by page max.
- This is a static prompt-only approximation of heavy-hitter ranking, not full decode-time H2O.

#### `pagekv_recon_static`
- Query-agnostic reconstruction proxy inside the `clusterkv` experiment path.
- Scores a page by the variance of keys inside the page around the page mean.
- This acts as a lightweight reconstruction-error-style proxy rather than a full KVZip implementation.

#### `pagekv_expected_attention_static`
- Static Expected Attention approximation inside the `clusterkv` experiment path.
- Estimates a query distribution from prompt queries only:
  - compute query mean and diagonal variance,
  - score each token with a diagonal-Gaussian expected-attention approximation,
  - aggregate token scores to pages by page max.
- This is not yet the full paper-faithful Expected Attention method because it does not explicitly model future RoPE positions and does not re-score during decoding.

#### `pagekv_random_static`
- Random page ranking baseline.
- Useful as a sanity-check floor for the static `clusterkv` scaffold.

### `tokenkv_*_static`
These ablations run through the same static ranking scaffold with `selection_granularity = token`.

Shared structure:
- score individual prefix tokens with one of the ranking backends below,
- keep the top-scoring tokens from the prefix plus the recent window.

Shared config fields:
- `window_size`
- `max_capacity_prompt`
- `ranking_backend`
- `observation_window`
- `selection_granularity = token`

Current token-level static ranking ablations:

#### `tokenkv_quest_bounds_static`
- Uses the current query to score each prefix token directly with `q · k`.
- This is the token-level counterpart of the page-level Quest-bound scorer.

#### `tokenkv_snapkv_static`
- Uses the last `observation_window` prompt queries, computes attention to prefix tokens, and averages those prompt-time attention weights directly at token granularity.

#### `tokenkv_h2o_static`
- Uses uniformly sampled prompt queries, computes attention to prefix tokens, and accumulates those attention weights directly at token granularity.

#### `tokenkv_recon_static`
- Query-agnostic reconstruction proxy at token granularity.
- Scores a token by squared deviation from the global prefix mean key.

#### `tokenkv_expected_attention_static`
- Static Expected Attention approximation at token granularity.
- Uses prompt-query mean and diagonal variance to score tokens with a diagonal-Gaussian expected-attention approximation.

#### `tokenkv_random_static`
- Random token ranking baseline.

### `clusterkv_*_static`
- The older `clusterkv_*_static` names are kept only as compatibility aliases in the experiment entrypoints.
- They currently point to the page-level static family rather than a true semantic-cluster implementation.

### `h2o_static`
- A stale experiment entry remains in [`test.py`](./test.py), but the repository does not currently contain `pred_h2o.py`.
- As a result, the dedicated `h2o` Modal method is not runnable in the current workspace without adding that missing runner.
- The nearest runnable H2O-style approximations in the current code are `pagekv_h2o_static` and `tokenkv_h2o_static`.

### LongBench / Modal experiment entrypoints
The Modal experiment file [`test.py`](./test.py) contains ready-to-run entrypoints for:
- `main_baseline`
- `main_snapkv_static`
- `main_quest_static`
- `main_pagekv_quest_bounds_static`
- `main_pagekv_snapkv_static`
- `main_pagekv_h2o_static`
- `main_pagekv_recon_static`
- `main_pagekv_expected_attention_static`
- `main_pagekv_random_static`
- `main_tokenkv_quest_bounds_static`
- `main_tokenkv_snapkv_static`
- `main_tokenkv_h2o_static`
- `main_tokenkv_recon_static`
- `main_tokenkv_expected_attention_static`
- `main_tokenkv_random_static`

Matching `main_eval_*` entrypoints score completed runs and `main_csv` aggregates saved results.

## TODO
- [ ] Add observation experiments for reduplication.
- [ ] Add LongBench for reduplication.
- [ ] Explore the prompt phase compression.

## Citation
If you feel this project is helpful, please consider cite our report :blush:
```
@article{li2024snapkv,
  title={SnapKV: LLM Knows What You are Looking for Before Generation},
  author={Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and Locatelli, Acyr and Ye, Hanchen and Cai, Tianle and Lewis, Patrick and Chen, Deming},
  journal={arXiv preprint arXiv:2404.14469},
  year={2024}
}
```
