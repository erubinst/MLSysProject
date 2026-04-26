# KV Importance Ranking Methods

This note summarizes the ranking methods we want to try for KV importance, with references to the original papers.

## Overview

These methods mainly differ along three axes:

- What signal they use to score importance
- When they score it
- Whether they are query-aware or query-agnostic

| Method | Ranking signal | Query-aware? | When applied | Main tradeoff |
|---|---|---:|---|---|
| H2O / Heavy-Hitters | Accumulated historical attention | Partly, via observed past queries | During decode | Simple online eviction, but depends on already-seen attention |
| SnapKV / high-attention prefill | High attention from a late prompt observation window, then cluster nearby tokens | Weakly / indirectly | Prefill / prompt compression | Fast and practical, but assumes prompt-phase attention predicts decode needs |
| KVZip / reconstruction error | Reconstruction error if a KV is removed or compressed | No, query-agnostic | Prefill / cache compression | More principled for reusable compression, but more expensive to score |
| Expected Attention | Closed-form expected future attention under a query distribution | Yes, statistically | Prefill and decode | More principled ranking, but more modeling complexity |

## 1. H2O: Heavy-Hitters

- Core idea: some tokens repeatedly attract large attention mass; keep those heavy hitters plus recent tokens.
- Signal: cumulative attention scores observed so far.
- Strength: online, simple, effective throughput improvement.
- Weakness: it can miss tokens that are not important yet but become important later.

Reference:

- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models, NeurIPS 2023
- https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html
- https://huggingface.co/papers/2306.14048

## 2. High-Attention Prefill: SnapKV

- Core idea: use the last part of the prompt as an observation window, find prompt positions receiving high attention from it, and keep those positions plus nearby clustered tokens.
- Signal: prompt-phase attention, usually pooled around salient positions.
- Strength: practical for prompt compression; avoids per-step rescoring.
- Weakness: mostly a prefill-time heuristic, so it can be wrong if decode-time needs differ from what the observation window suggested.

Reference:

- SnapKV: LLM Knows What You are Looking for Before Generation, NeurIPS 2024
- https://openreview.net/forum?id=poE54GOq2l
- https://huggingface.co/papers/2404.14469

## 3. Alternative Clustering for SnapKV-Style Pipelines

These are not standalone KV ranking methods. They are alternative clustering procedures for already-selected salient tokens or features.

### k-means++

- Core idea: improve centroid initialization for k-means.
- Strength: better stability and fewer bad seeds than vanilla random initialization.
- Weakness: still standard Euclidean k-means; not inherently aligned with cosine geometry.

Reference:

- k-means++: The Advantages of Careful Seeding
- https://research.google/pubs/k-means-the-advantages-of-careful-seeding/
- https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

### spherical k-means

- Core idea: run k-means on normalized vectors using cosine similarity rather than Euclidean distance.
- Strength: often better when direction matters more than magnitude, which is common for embeddings and attention features.
- Weakness: still a clustering heuristic rather than a direct notion of token importance.

Reference:

- Dhillon and Modha, Concept Decompositions for Large Sparse Text Data Using Clustering
- https://doi.org/10.1023/A:1007612920971
- Background exposition: https://www.jstatsoft.org/article/view/v050i10

## 4. KVZip: Reconstruction Error

- Core idea: compress or evict KV pairs based on how much they matter for reconstructing the original context representation.
- Signal: reconstruction loss or reconstruction error, not attention.
- Strength: query-agnostic and reusable across future queries.
- Weakness: scoring is usually more expensive than simple attention heuristics.

Reference:

- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
- https://huggingface.co/papers/2505.23416
- https://arxiv.gg/abs/2505.23416

## 5. Expected Attention

- Core idea: instead of waiting to see future queries, estimate their distribution and compute expected attention analytically.
- Signal: estimated future attention in closed form.
- Strength: directly aligned with future decode-time importance without needing actual future queries.
- Weakness: newer and more complex; depends on the quality of the approximation to the future-query distribution.

Reference:

- Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution
- https://huggingface.co/papers/2510.00636

## Practical Interpretation

- If we want cheap online pruning, H2O is the baseline.
- If we want prompt-only compression with low engineering cost, SnapKV-style high-attention prefill is attractive.
- If we want reusable compressed caches across multiple downstream queries, KVZip is conceptually strongest of this list.
- If we want the most Quest-like or future-query-aware direction, Expected Attention is the closest in spirit.

## Clean Taxonomy

- Historical attention: H2O
- Prompt observation attention: SnapKV
- Representation fidelity: KVZip
- Predicted future attention: Expected Attention
