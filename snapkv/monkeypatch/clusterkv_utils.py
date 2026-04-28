import torch
from typing import Optional
import math

class OnlineKMeans:
    def __init__(self, n_clusters: int, dim: int, device: torch.device):
        self.n_clusters = n_clusters
        self.dim = dim
        self.device = device
        self.centroids = None
        self.cluster_sizes = None
        self.initialized = False

    def initialize(self, data: torch.Tensor):
        """Initialize centroids using k-means++ like initialization"""
        n_samples = data.shape[0]
        centroids = torch.zeros(self.n_clusters, self.dim, device=self.device, dtype=data.dtype)

        # First centroid: random selection
        idx = torch.randint(0, n_samples, (1,), device=self.device)
        centroids[0] = data[idx]

        # Remaining centroids: k-means++ initialization
        for i in range(1, self.n_clusters):
            distances = torch.cdist(data, centroids[:i])
            min_distances = distances.min(dim=1)[0]
            probs = min_distances / min_distances.sum()
            idx = torch.multinomial(probs, 1)
            centroids[i] = data[idx]

        self.centroids = centroids
        self.cluster_sizes = torch.ones(self.n_clusters, device=self.device, dtype=torch.float32)
        self.initialized = True

    def update(self, data: torch.Tensor):
        """Online update of centroids"""
        if not self.initialized:
            self.initialize(data)
            return

        # Find closest centroids
        distances = torch.cdist(data, self.centroids)
        closest_clusters = distances.argmin(dim=1)

        # Update centroids and sizes
        for i in range(self.n_clusters):
            mask = (closest_clusters == i)
            if mask.any():
                new_points = data[mask]
                n_new = new_points.shape[0]

                # Online update formula: centroid = (size * centroid + sum(new_points)) / (size + n_new)
                old_size = self.cluster_sizes[i]
                new_size = old_size + n_new

                self.centroids[i] = (old_size * self.centroids[i] + new_points.sum(dim=0)) / new_size
                self.cluster_sizes[i] = new_size

    def quantize_centroids_int8(self):
        """Quantize centroids to INT8"""
        if self.centroids is None:
            return None

        # Simple quantization: scale to [-128, 127]
        centroids_min = self.centroids.min(dim=0, keepdim=True)[0]
        centroids_max = self.centroids.max(dim=0, keepdim=True)[0]

        # Avoid division by zero
        scale = (centroids_max - centroids_min) / 255.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        # Quantize
        quantized = ((self.centroids - centroids_min) / scale - 128).clamp(-128, 127).to(torch.int8)

        return quantized, centroids_min, scale

    def get_top_k_clusters(self, query: torch.Tensor, k: int):
        """Get top-K closest clusters for a query"""
        if not self.initialized:
            return None

        distances = torch.cdist(query.unsqueeze(0), self.centroids)
        top_k_indices = distances.topk(k, dim=1, largest=False).indices.squeeze(0)

        return top_k_indices

class ClusterKVCache:
    def __init__(self, n_clusters: int = 128, window_size: int = 64, max_capacity_prompt: int = 4096,
                 update_policy: str = 'incremental', update_interval: int = 100,
                 page_size: int = 16, ranking_backend: str = 'quest_bounds',
                 observation_window: int = 32, selection_granularity: str = 'page',
                 clustering_backend: str = 'kmeanspp',
                 num_block: int = 12,
                 theta: float = 0.0,
                 n_future_positions: int = 512,
                 n_sink: int = 4,
                 use_covariance: bool = True,
                 use_vnorm: bool = True,
                 epsilon: float = 0.0,
                 hidden_states_buffer_size: int = 128):
        """
        Args:
            n_clusters: Number of clusters for k-means
            window_size: Size of recent window to keep uncompressed
            max_capacity_prompt: Maximum capacity for prompt phase
            update_policy: 'static', 'incremental', or 'periodic'
            update_interval: For periodic updates, update every N tokens
            page_size: Number of KV tokens grouped into one retrieval page
            ranking_backend: Importance ranking backend for prefix pages
            observation_window: Number of prompt queries used by sampled attention backends
            selection_granularity: 'page', 'token', 'cluster', or 'clusterattn'
            clustering_backend: 'kmeans', 'kmeanspp', or 'spherical_kmeans'
            num_block: ClusterAttn-style number of blocks for density selection
            theta: ClusterAttn-style threshold for block centers
        """
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.update_policy = update_policy
        self.update_interval = update_interval
        self.page_size = page_size
        self.ranking_backend = ranking_backend
        self.observation_window = observation_window
        self.selection_granularity = selection_granularity
        # Some configs may set clustering_backend=None; treat it as default.
        self.clustering_backend = clustering_backend or "kmeanspp"
        self.num_block = num_block
        self.theta = theta
        self.n_future_positions = n_future_positions
        self.n_sink = n_sink
        self.use_covariance = use_covariance
        self.use_vnorm = use_vnorm
        self.epsilon = epsilon
        self.hidden_states_buffer_size = hidden_states_buffer_size

        self.kmeans = None
        self.cluster_assignments = None
        self.compressed_keys = None
        self.compressed_values = None
        self.tokens_processed = 0
        self.last_refresh_total_len = 0
        self.expected_query_buffer = None
        self.active_pre_rope_query_states = None
        self.active_layer_module = None
        self.active_total_len = 0

    def reset(self):
        """Reset the cache"""
        self.kmeans = None
        self.cluster_assignments = None
        self.compressed_keys = None
        self.compressed_values = None
        self.tokens_processed = 0
        self.last_refresh_total_len = 0
        self.expected_query_buffer = None
        self.active_pre_rope_query_states = None
        self.active_layer_module = None
        self.active_total_len = 0

    def _append_expected_query_buffer(self, pre_rope_query_states: torch.Tensor, total_len: int):
        if pre_rope_query_states is None:
            return

        if pre_rope_query_states.shape[2] > 1 or total_len == pre_rope_query_states.shape[2]:
            buffered = pre_rope_query_states
        else:
            new_states = pre_rope_query_states[:, :, -1:, :]
            if self.expected_query_buffer is None:
                buffered = new_states
            else:
                buffered = torch.cat([self.expected_query_buffer, new_states], dim=2)

        if self.hidden_states_buffer_size > 0 and buffered.shape[2] > self.hidden_states_buffer_size:
            buffered = buffered[:, :, -self.hidden_states_buffer_size :, :]
        self.expected_query_buffer = buffered

    def _rotate_mean_and_var_over_future_positions(self, mu: torch.Tensor, var: torch.Tensor):
        module = self.active_layer_module
        if module is None:
            return mu, var

        head_dim = mu.shape[-1]
        half_dim = head_dim // 2
        if half_dim == 0:
            return mu, var

        future_start = int(self.active_total_len)
        future_end = future_start + max(1, int(self.n_future_positions))

        dummy = mu.new_zeros((mu.shape[0], 1, 1, head_dim))
        cos, sin = module.rotary_emb(dummy, seq_len=future_end)

        while cos.dim() > 2:
            cos = cos[0]
            sin = sin[0]

        cos = cos[future_start:future_end]
        sin = sin[future_start:future_end]

        if cos.shape[-1] != half_dim:
            cos = cos[..., :half_dim]
            sin = sin[..., :half_dim]

        mu1, mu2 = mu[..., :half_dim], mu[..., half_dim : 2 * half_dim]
        var1, var2 = var[..., :half_dim], var[..., half_dim : 2 * half_dim]

        cos = cos.to(mu.dtype).unsqueeze(0).unsqueeze(0)
        sin = sin.to(mu.dtype).unsqueeze(0).unsqueeze(0)

        mu1_rot = mu1.unsqueeze(2) * cos - mu2.unsqueeze(2) * sin
        mu2_rot = mu1.unsqueeze(2) * sin + mu2.unsqueeze(2) * cos
        mu_rot = torch.cat([mu1_rot, mu2_rot], dim=-1).mean(dim=2)

        if head_dim > 2 * half_dim:
            mu_rot = torch.cat([mu_rot, mu[..., 2 * half_dim :]], dim=-1)

        cos2 = cos.pow(2)
        sin2 = sin.pow(2)
        var1_rot = var1.unsqueeze(2) * cos2 + var2.unsqueeze(2) * sin2
        var2_rot = var1.unsqueeze(2) * sin2 + var2.unsqueeze(2) * cos2
        var_rot = torch.cat([var1_rot, var2_rot], dim=-1).mean(dim=2)

        if head_dim > 2 * half_dim:
            var_rot = torch.cat([var_rot, var[..., 2 * half_dim :].unsqueeze(2).expand(-1, -1, var_rot.shape[2], -1)], dim=-1)
            var_rot = var_rot.mean(dim=2)
        else:
            var_rot = var_rot

        return mu_rot, var_rot

    def _score_tokens_expected_attention_paper(self, prefix_keys: torch.Tensor, prefix_values: torch.Tensor):
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        if prefix_len == 0:
            return prefix_keys.new_empty((bsz, num_heads, 0))

        pre_rope_query_states = self.active_pre_rope_query_states
        if pre_rope_query_states is None or self.active_layer_module is None:
            fallback_queries = prefix_keys.new_zeros((bsz, num_heads, 1, head_dim))
            return self._score_tokens_expected_attention_closed_form(prefix_keys, fallback_queries)

        query_states = pre_rope_query_states
        if query_states.shape[2] > self.n_sink:
            query_states = query_states[:, :, self.n_sink :, :]

        if query_states.shape[2] == 0:
            query_states = pre_rope_query_states

        mu = query_states.mean(dim=2)
        if self.use_covariance:
            var = query_states.var(dim=2, unbiased=False)
        else:
            var = torch.zeros_like(mu)

        mu, var = self._rotate_mean_and_var_over_future_positions(mu, var)

        token_scores = torch.einsum("bhd,bhkd->bhk", mu, prefix_keys) / math.sqrt(head_dim)
        if self.use_covariance:
            token_scores = token_scores + 0.5 * torch.einsum("bhd,bhkd->bhk", var, prefix_keys.pow(2)) / head_dim

        token_scores = torch.softmax(token_scores, dim=-1)

        if self.use_vnorm:
            token_scores = (token_scores + self.epsilon) * prefix_values.norm(dim=-1)

        if prefix_len > 0 and self.n_sink > 0:
            keep = min(self.n_sink, prefix_len)
            sink_value = token_scores.max(dim=-1, keepdim=True).values
            token_scores[:, :, :keep] = sink_value

        return token_scores

    def _score_tokens_expected_attention_closed_form(self, prefix_keys: torch.Tensor, query_states: torch.Tensor):
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        if prefix_len == 0:
            return prefix_keys.new_empty((bsz, num_heads, 0))

        sampled_queries = self._sample_queries(query_states, mode="uniform")
        mu = sampled_queries.mean(dim=2)
        var = sampled_queries.var(dim=2, unbiased=False) if sampled_queries.shape[2] > 1 else torch.zeros_like(mu)

        token_scores = torch.einsum("bhd,bhkd->bhk", mu, prefix_keys) / math.sqrt(head_dim)
        if self.use_covariance:
            token_scores = token_scores + 0.5 * torch.einsum("bhd,bhkd->bhk", var, prefix_keys.pow(2)) / head_dim

        token_scores = torch.softmax(token_scores, dim=-1)
        return (token_scores + self.epsilon) * prefix_keys.norm(dim=-1) if self.use_vnorm else token_scores

    def _pad_to_pages(self, tensor: torch.Tensor):
        bsz, num_heads, prefix_len, head_dim = tensor.shape
        if prefix_len == 0:
            empty_pages = tensor.new_empty((bsz, num_heads, 0, self.page_size, head_dim))
            return empty_pages, prefix_len, 0

        pad_len = (self.page_size - (prefix_len % self.page_size)) % self.page_size
        if pad_len:
            tensor = torch.cat([tensor, tensor.new_zeros(bsz, num_heads, pad_len, head_dim)], dim=2)

        num_pages = tensor.shape[2] // self.page_size
        pages = tensor.reshape(bsz, num_heads, num_pages, self.page_size, head_dim)
        return pages, prefix_len, pad_len

    def _sample_queries(self, query_states: torch.Tensor, mode: str):
        q_len = query_states.shape[2]
        if q_len == 0:
            return query_states

        sample_count = min(self.observation_window, q_len)
        if sample_count <= 0 or sample_count == q_len:
            return query_states

        if mode == "tail":
            indices = torch.arange(q_len - sample_count, q_len, device=query_states.device)
        elif mode == "uniform":
            indices = torch.linspace(0, q_len - 1, steps=sample_count, device=query_states.device)
            indices = indices.round().long().unique(sorted=True)
        else:
            raise ValueError(f"Unknown query sampling mode: {mode}")

        return query_states.index_select(2, indices)

    def _aggregate_token_scores_to_pages(self, token_scores: torch.Tensor, prefix_len: int):
        if prefix_len == 0:
            return token_scores.new_empty(token_scores.shape[0], token_scores.shape[1], 0)

        pad_len = (self.page_size - (prefix_len % self.page_size)) % self.page_size
        if pad_len:
            token_scores = torch.cat(
                [token_scores, token_scores.new_full((token_scores.shape[0], token_scores.shape[1], pad_len), float("-inf"))],
                dim=-1,
            )

        page_scores = token_scores.reshape(token_scores.shape[0], token_scores.shape[1], -1, self.page_size)
        return page_scores.amax(dim=-1)

    def _run_kmeans_single(self, data: torch.Tensor):
        """
        Lightweight static k-means for one (batch, head) prefix slice.
        Returns assignments and centroids.
        """
        n_tokens, head_dim = data.shape
        if n_tokens == 0:
            return data.new_empty((0,), dtype=torch.long), data.new_empty((0, head_dim))

        k = min(self.n_clusters, n_tokens)
        spherical = self.clustering_backend == "spherical_kmeans"
        if spherical:
            work_data = torch.nn.functional.normalize(data, dim=-1)
        else:
            work_data = data

        centroids = torch.empty((k, head_dim), device=data.device, dtype=data.dtype)

        if self.clustering_backend == "kmeans":
            init_indices = torch.randperm(n_tokens, device=data.device)[:k]
            centroids.copy_(work_data[init_indices])
        elif self.clustering_backend in ("kmeanspp", "spherical_kmeans"):
            first_idx = torch.randint(0, n_tokens, (1,), device=data.device)
            centroids[0] = work_data[first_idx]
            # Compute k-means++ distances in fp32 for numerical stability.
            # In fp16, squared distances can overflow -> NaNs/Inf -> multinomial failures.
            closest_dist_sq = ((work_data.float() - centroids[0:1].float()) ** 2).sum(dim=-1)
            for i in range(1, k):
                total = closest_dist_sq.sum()
                if (not torch.isfinite(total)) or total <= 0:
                    next_idx = torch.randint(0, n_tokens, (1,), device=data.device)
                else:
                    probs = closest_dist_sq / total
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    probs = probs.clamp_min_(0.0)
                    s = probs.sum()
                    if (not torch.isfinite(s)) or s <= 0:
                        next_idx = torch.randint(0, n_tokens, (1,), device=data.device)
                    else:
                        probs = probs / s
                        next_idx = torch.multinomial(probs, 1)
                centroids[i] = work_data[next_idx]
                new_dist_sq = ((work_data.float() - centroids[i : i + 1].float()) ** 2).sum(dim=-1)
                closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)
        else:
            raise ValueError(f"Unsupported clustering backend: {self.clustering_backend}")

        assignments = None
        for _ in range(8):
            if spherical:
                similarities = work_data @ centroids.T
                new_assignments = similarities.argmax(dim=-1)
            else:
                distances = torch.cdist(work_data, centroids)
                new_assignments = distances.argmin(dim=-1)
            if assignments is not None and torch.equal(new_assignments, assignments):
                break
            assignments = new_assignments

            for cluster_id in range(k):
                mask = assignments == cluster_id
                if mask.any():
                    centroids[cluster_id] = work_data[mask].mean(dim=0)
                    if spherical:
                        centroids[cluster_id] = torch.nn.functional.normalize(centroids[cluster_id], dim=-1)

        return assignments, centroids

    def _cluster_prefix(self, prefix_keys: torch.Tensor):
        """
        Cluster prefix tokens independently per batch/head.
        """
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        assignments = torch.empty((bsz, num_heads, prefix_len), device=prefix_keys.device, dtype=torch.long)
        centroids = []
        cluster_counts = torch.empty((bsz, num_heads), device=prefix_keys.device, dtype=torch.long)

        for b in range(bsz):
            head_centroids = []
            for h in range(num_heads):
                assign_bh, centroids_bh = self._run_kmeans_single(prefix_keys[b, h])
                assignments[b, h] = assign_bh
                cluster_counts[b, h] = centroids_bh.shape[0]
                head_centroids.append(centroids_bh)
            centroids.append(head_centroids)

        return assignments, centroids, cluster_counts

    def _score_pages_quest_bounds(self, prefix_keys: torch.Tensor, current_query: torch.Tensor):
        """
        Estimate page criticality with Quest-style upper bounds.

        Args:
            prefix_keys: [bsz, num_heads, prefix_len, head_dim]
            current_query: [bsz, num_heads, head_dim]
        Returns:
            page_scores: [bsz, num_heads, num_pages]
            page_keys: [bsz, num_heads, num_pages, page_size, head_dim]
            prefix_len: original prefix length before padding
        """
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        if prefix_len == 0:
            empty_scores = prefix_keys.new_empty((bsz, num_heads, 0))
            empty_pages = prefix_keys.new_empty((bsz, num_heads, 0, self.page_size, head_dim))
            return empty_scores, empty_pages, prefix_len

        page_keys, _, _ = self._pad_to_pages(prefix_keys)
        page_min = page_keys.amin(dim=3)
        page_max = page_keys.amax(dim=3)

        query = current_query.unsqueeze(2)
        page_scores = torch.maximum(query * page_min, query * page_max).sum(dim=-1)
        return page_scores, page_keys, prefix_len

    def _score_tokens_quest_bounds(self, prefix_keys: torch.Tensor, current_query: torch.Tensor):
        prefix_len = prefix_keys.shape[2]
        if prefix_len == 0:
            return prefix_keys.new_empty((prefix_keys.shape[0], prefix_keys.shape[1], 0))
        return torch.einsum("bhd,bhtd->bht", current_query, prefix_keys)

    def _score_pages_prefill_attention(self, prefix_keys: torch.Tensor, query_states: torch.Tensor):
        bsz, num_heads, prefix_len, _ = prefix_keys.shape
        if prefix_len == 0:
            empty_scores = prefix_keys.new_empty((bsz, num_heads, 0))
            page_keys, _, _ = self._pad_to_pages(prefix_keys)
            return empty_scores, page_keys, prefix_len

        sampled_queries = self._sample_queries(query_states, mode="tail")
        attn_logits = torch.einsum("bhqd,bhkd->bhqk", sampled_queries, prefix_keys) / math.sqrt(prefix_keys.shape[-1])
        token_scores = torch.softmax(attn_logits, dim=-1).mean(dim=2)
        page_scores = self._aggregate_token_scores_to_pages(token_scores, prefix_len)
        page_keys, _, _ = self._pad_to_pages(prefix_keys)
        return page_scores, page_keys, prefix_len

    def _score_tokens_prefill_attention(self, prefix_keys: torch.Tensor, query_states: torch.Tensor):
        bsz, num_heads, prefix_len, _ = prefix_keys.shape
        if prefix_len == 0:
            return prefix_keys.new_empty((bsz, num_heads, 0))

        sampled_queries = self._sample_queries(query_states, mode="tail")
        attn_logits = torch.einsum("bhqd,bhkd->bhqk", sampled_queries, prefix_keys) / math.sqrt(prefix_keys.shape[-1])
        return torch.softmax(attn_logits, dim=-1).mean(dim=2)

    def _score_pages_h2o(self, prefix_keys: torch.Tensor, query_states: torch.Tensor):
        bsz, num_heads, prefix_len, _ = prefix_keys.shape
        if prefix_len == 0:
            empty_scores = prefix_keys.new_empty((bsz, num_heads, 0))
            page_keys, _, _ = self._pad_to_pages(prefix_keys)
            return empty_scores, page_keys, prefix_len

        sampled_queries = self._sample_queries(query_states, mode="uniform")
        attn_logits = torch.einsum("bhqd,bhkd->bhqk", sampled_queries, prefix_keys) / math.sqrt(prefix_keys.shape[-1])
        token_scores = torch.softmax(attn_logits, dim=-1).sum(dim=2)
        page_scores = self._aggregate_token_scores_to_pages(token_scores, prefix_len)
        page_keys, _, _ = self._pad_to_pages(prefix_keys)
        return page_scores, page_keys, prefix_len

    def _score_tokens_h2o(self, prefix_keys: torch.Tensor, query_states: torch.Tensor):
        bsz, num_heads, prefix_len, _ = prefix_keys.shape
        if prefix_len == 0:
            return prefix_keys.new_empty((bsz, num_heads, 0))

        sampled_queries = self._sample_queries(query_states, mode="uniform")
        attn_logits = torch.einsum("bhqd,bhkd->bhqk", sampled_queries, prefix_keys) / math.sqrt(prefix_keys.shape[-1])
        return torch.softmax(attn_logits, dim=-1).sum(dim=2)

    def _score_pages_reconstruction(self, prefix_keys: torch.Tensor):
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        if prefix_len == 0:
            empty_scores = prefix_keys.new_empty((bsz, num_heads, 0))
            empty_pages = prefix_keys.new_empty((bsz, num_heads, 0, self.page_size, head_dim))
            return empty_scores, empty_pages, prefix_len

        page_keys, _, _ = self._pad_to_pages(prefix_keys)
        page_center = page_keys.mean(dim=3, keepdim=True)
        page_scores = (page_keys - page_center).pow(2).sum(dim=(3, 4))
        return page_scores, page_keys, prefix_len

    def _score_tokens_reconstruction(self, prefix_keys: torch.Tensor):
        bsz, num_heads, prefix_len, _ = prefix_keys.shape
        if prefix_len == 0:
            return prefix_keys.new_empty((bsz, num_heads, 0))

        global_center = prefix_keys.mean(dim=2, keepdim=True)
        return (prefix_keys - global_center).pow(2).sum(dim=-1)

    def _score_pages_expected_attention(self, prefix_keys: torch.Tensor, query_states: torch.Tensor):
        """
        Static Expected Attention approximation.

        We estimate a future-query distribution from prompt queries only and use a
        diagonal-Gaussian closed-form approximation:
            E[exp(q·k / sqrt(d))] ~= exp(mu·k / sqrt(d) + 0.5 * (var·k^2) / d)

        This is a practical static scorer for the current prompt-time compression
        path. It does not model future RoPE positions explicitly.
        """
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        if prefix_len == 0:
            empty_scores = prefix_keys.new_empty((bsz, num_heads, 0))
            empty_pages = prefix_keys.new_empty((bsz, num_heads, 0, self.page_size, head_dim))
            return empty_scores, empty_pages, prefix_len

        token_scores = self._score_tokens_expected_attention_paper(prefix_keys, prefix_keys)

        page_scores = self._aggregate_token_scores_to_pages(token_scores, prefix_len)
        page_keys, _, _ = self._pad_to_pages(prefix_keys)
        return page_scores, page_keys, prefix_len

    def _score_tokens_expected_attention(self, prefix_keys: torch.Tensor, query_states: torch.Tensor):
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        if prefix_len == 0:
            return prefix_keys.new_empty((bsz, num_heads, 0))

        return self._score_tokens_expected_attention_closed_form(prefix_keys, query_states)

    def _select_clusterattn_density(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        current_query: torch.Tensor,
        query_states: torch.Tensor,
    ):
        """
        ClusterAttn (Algorithm 1) inspired selection on *token indices*:
        - build token importance P using the configured token ranking backend
        - aggregate to block scores P' via maxpool with stride=blksize
        - filter blocks by theta to get centers
        - expand centers by radius r in block units to get candidate blocks
        - select top token_budget tokens within candidate blocks, keep chronological order
        """
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        if prefix_len == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        token_budget = max(self.max_capacity_prompt - self.window_size, 0)
        token_budget = min(token_budget, prefix_len)
        if token_budget == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        # P: per-token importance from the configured backend.
        token_scores = self._score_tokens(prefix_keys, current_query, query_states)  # [bsz, heads, prefix_len]

        # Block maxpool -> P' (block scores)
        num_block = int(max(1, self.num_block))
        blksize = int(max(1, math.ceil(prefix_len / float(num_block))))
        nblocks = int(math.ceil(prefix_len / float(blksize)))

        x = token_scores.reshape(bsz * num_heads, 1, prefix_len)
        pad = nblocks * blksize - prefix_len
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad), value=torch.finfo(token_scores.dtype).min)
        p_block = torch.nn.functional.max_pool1d(x, kernel_size=blksize, stride=blksize)
        p_block = p_block.reshape(bsz, num_heads, nblocks)  # [bsz, heads, nblocks]

        centers_mask = p_block >= float(self.theta)
        centers_count = centers_mask.sum(dim=-1)  # [bsz, heads]

        # If no centers found, fall back to vanilla top-k on tokens
        if (centers_count == 0).any():
            top_idx = token_scores.topk(token_budget, dim=-1).indices.sort(dim=-1).values
            gather_index = top_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            return (
                prefix_keys.gather(dim=2, index=gather_index),
                prefix_values.gather(dim=2, index=gather_index),
            )

        # Neighborhood radius r in block units (token_budget translated to ~blocks).
        # Use a per-(batch, head) loop to avoid ragged tensor issues.
        budget_blocks = max(1, int(math.ceil(token_budget / float(blksize))))
        r = torch.clamp((budget_blocks / (2.0 * centers_count.float())).floor().to(torch.long), min=0)  # [bsz, heads]

        cand_blocks = torch.zeros((bsz, num_heads, nblocks), device=prefix_keys.device, dtype=torch.bool)
        for b in range(bsz):
            for h in range(num_heads):
                centers = torch.nonzero(centers_mask[b, h], as_tuple=False).squeeze(-1)
                if centers.numel() == 0:
                    continue
                r_bh = int(r[b, h].item())
                if r_bh == 0:
                    cand_blocks[b, h, centers] = True
                    continue
                offsets = torch.arange(-r_bh, r_bh + 1, device=prefix_keys.device, dtype=torch.long)
                expanded = (centers.unsqueeze(-1) + offsets.view(1, -1)).clamp_(0, nblocks - 1).reshape(-1)
                cand_blocks[b, h, expanded.unique()] = True

        tok_ids = torch.arange(prefix_len, device=prefix_keys.device).view(1, 1, -1).expand(bsz, num_heads, -1)
        tok_block = (tok_ids // blksize).clamp_max(nblocks - 1)
        tok_mask = cand_blocks.gather(-1, tok_block)  # [bsz, heads, prefix_len]

        neg_inf = torch.finfo(token_scores.dtype).min
        masked_scores = token_scores.masked_fill(~tok_mask, neg_inf)
        idx = masked_scores.topk(token_budget, dim=-1).indices
        vals = masked_scores.gather(-1, idx)
        need_fill = vals.eq(neg_inf)
        if need_fill.any():
            outside_scores = token_scores.masked_fill(tok_mask, neg_inf)
            outside = outside_scores.topk(token_budget, dim=-1).indices
            idx = torch.where(need_fill, outside, idx)

        idx = idx.sort(dim=-1).values
        gather_index = idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        return (
            prefix_keys.gather(dim=2, index=gather_index),
            prefix_values.gather(dim=2, index=gather_index),
        )

    def _score_pages_random(self, prefix_keys: torch.Tensor):
        bsz, num_heads, prefix_len, head_dim = prefix_keys.shape
        if prefix_len == 0:
            empty_scores = prefix_keys.new_empty((bsz, num_heads, 0))
            empty_pages = prefix_keys.new_empty((bsz, num_heads, 0, self.page_size, head_dim))
            return empty_scores, empty_pages, prefix_len

        page_keys, _, _ = self._pad_to_pages(prefix_keys)
        page_scores = torch.rand(
            (bsz, num_heads, page_keys.shape[2]),
            device=prefix_keys.device,
            dtype=prefix_keys.dtype,
        )
        return page_scores, page_keys, prefix_len

    def _score_tokens_random(self, prefix_keys: torch.Tensor):
        bsz, num_heads, prefix_len, _ = prefix_keys.shape
        return torch.rand((bsz, num_heads, prefix_len), device=prefix_keys.device, dtype=prefix_keys.dtype)

    def _score_pages(self, prefix_keys: torch.Tensor, current_query: torch.Tensor, query_states: torch.Tensor):
        if self.ranking_backend == "quest_bounds":
            return self._score_pages_quest_bounds(prefix_keys, current_query)
        if self.ranking_backend == "snapkv_prefill":
            return self._score_pages_prefill_attention(prefix_keys, query_states)
        if self.ranking_backend == "h2o_accum":
            return self._score_pages_h2o(prefix_keys, query_states)
        if self.ranking_backend == "reconstruction_error":
            return self._score_pages_reconstruction(prefix_keys)
        if self.ranking_backend == "expected_attention":
            return self._score_pages_expected_attention(prefix_keys, query_states)
        if self.ranking_backend == "random":
            return self._score_pages_random(prefix_keys)
        raise ValueError(f"Unsupported ranking backend: {self.ranking_backend}")

    def _score_tokens(self, prefix_keys: torch.Tensor, current_query: torch.Tensor, query_states: torch.Tensor):
        if self.ranking_backend == "quest_bounds":
            return self._score_tokens_quest_bounds(prefix_keys, current_query)
        if self.ranking_backend == "snapkv_prefill":
            return self._score_tokens_prefill_attention(prefix_keys, query_states)
        if self.ranking_backend == "h2o_accum":
            return self._score_tokens_h2o(prefix_keys, query_states)
        if self.ranking_backend == "reconstruction_error":
            return self._score_tokens_reconstruction(prefix_keys)
        if self.ranking_backend == "expected_attention":
            return self._score_tokens_expected_attention(prefix_keys, query_states)
        if self.ranking_backend == "random":
            return self._score_tokens_random(prefix_keys)
        raise ValueError(f"Unsupported ranking backend: {self.ranking_backend}")

    def _select_topk_tokens(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        current_query: torch.Tensor,
        query_states: torch.Tensor,
    ):
        token_scores = self._score_tokens(prefix_keys, current_query, query_states)
        prefix_len = prefix_keys.shape[2]
        if prefix_len == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        token_budget = max(self.max_capacity_prompt - self.window_size, 0)
        token_budget = min(token_budget, prefix_len)
        if token_budget == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        head_dim = prefix_keys.shape[-1]
        top_token_indices = token_scores.topk(token_budget, dim=-1).indices
        top_token_indices = top_token_indices.sort(dim=-1).values
        gather_index = top_token_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        selected_keys = prefix_keys.gather(dim=2, index=gather_index)
        selected_values = prefix_values.gather(dim=2, index=gather_index)
        return selected_keys, selected_values

    def _select_topk_clusters(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        current_query: torch.Tensor,
        query_states: torch.Tensor,
    ):
        token_scores = self._score_tokens(prefix_keys, current_query, query_states)
        prefix_len = prefix_keys.shape[2]
        if prefix_len == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        token_budget = max(self.max_capacity_prompt - self.window_size, 0)
        token_budget = min(token_budget, prefix_len)
        if token_budget == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        assignments, _, cluster_counts = self._cluster_prefix(prefix_keys)
        bsz, num_heads, _, head_dim = prefix_keys.shape
        selected_keys = prefix_keys.new_empty((bsz, num_heads, token_budget, head_dim))
        selected_values = prefix_values.new_empty((bsz, num_heads, token_budget, head_dim))

        for b in range(bsz):
            for h in range(num_heads):
                scores_bh = token_scores[b, h]
                assign_bh = assignments[b, h]
                num_clusters = int(cluster_counts[b, h].item())

                cluster_scores = scores_bh.new_full((num_clusters,), float("-inf"))
                for cluster_id in range(num_clusters):
                    mask = assign_bh == cluster_id
                    if mask.any():
                        cluster_scores[cluster_id] = scores_bh[mask].amax()

                ranked_clusters = cluster_scores.argsort(descending=True)
                selected_indices = []
                remaining = token_budget
                for cluster_id in ranked_clusters.tolist():
                    member_indices = torch.nonzero(assign_bh == cluster_id, as_tuple=False).squeeze(-1)
                    if member_indices.numel() == 0:
                        continue

                    if member_indices.numel() <= remaining:
                        selected_indices.append(member_indices)
                        remaining -= member_indices.numel()
                    else:
                        member_scores = scores_bh[member_indices]
                        keep_local = member_scores.topk(remaining, dim=0).indices
                        kept = member_indices[keep_local].sort().values
                        selected_indices.append(kept)
                        remaining = 0

                    if remaining == 0:
                        break

                if selected_indices:
                    selected_indices = torch.cat(selected_indices).sort().values
                else:
                    selected_indices = torch.empty((0,), device=prefix_keys.device, dtype=torch.long)

                if selected_indices.numel() < token_budget:
                    filler = scores_bh.topk(token_budget, dim=0).indices.sort().values
                    selected_indices = filler

                gather_index = selected_indices.unsqueeze(-1).expand(-1, head_dim)
                selected_keys[b, h] = prefix_keys[b, h].gather(dim=0, index=gather_index)
                selected_values[b, h] = prefix_values[b, h].gather(dim=0, index=gather_index)

        return selected_keys, selected_values

    def _select_topk_pages(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        current_query: torch.Tensor,
        query_states: torch.Tensor,
    ):
        """Select top scoring pages from the prefix and preserve chronological order."""
        page_scores, page_keys, prefix_len = self._score_pages(prefix_keys, current_query, query_states)
        if prefix_len == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        _, _, _, _, head_dim = page_keys.shape
        pad_len = (self.page_size - (prefix_len % self.page_size)) % self.page_size
        if pad_len:
            prefix_values = torch.cat(
                [prefix_values, prefix_values.new_zeros(prefix_values.shape[0], prefix_values.shape[1], pad_len, head_dim)],
                dim=2,
            )
        page_values = prefix_values.reshape(prefix_values.shape[0], prefix_values.shape[1], -1, self.page_size, head_dim)

        token_budget = max(self.max_capacity_prompt - self.window_size, 0)
        if token_budget == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        num_pages = page_scores.shape[-1]
        pages_to_keep = min((token_budget + self.page_size - 1) // self.page_size, num_pages)
        if pages_to_keep == 0:
            return prefix_keys[:, :, :0, :], prefix_values[:, :, :0, :]

        top_page_indices = page_scores.topk(pages_to_keep, dim=-1).indices
        top_page_indices = top_page_indices.sort(dim=-1).values

        gather_index = top_page_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, self.page_size, head_dim
        )
        selected_keys = page_keys.gather(dim=2, index=gather_index).reshape(
            prefix_keys.shape[0], prefix_keys.shape[1], -1, head_dim
        )
        selected_values = page_values.gather(dim=2, index=gather_index).reshape(
            prefix_values.shape[0], prefix_values.shape[1], -1, head_dim
        )

        return selected_keys[:, :, :token_budget, :], selected_values[:, :, :token_budget, :]

    def update_kv(self, key_states: torch.Tensor, query_states: torch.Tensor,
                  value_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                  num_key_value_groups: int = 1, pre_rope_query_states: Optional[torch.Tensor] = None,
                  layer_module: Optional[object] = None, total_seq_len: Optional[int] = None):
        """
        Update KV cache using query-aware retrieval.
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        cache_len = key_states.shape[2]
        total_len = int(total_seq_len) if total_seq_len is not None else cache_len
        self._append_expected_query_buffer(pre_rope_query_states, total_len)

        # During prompt phase, keep all tokens if below capacity
        if cache_len < self.max_capacity_prompt:
            self.compressed_keys = None
            self.compressed_values = None
            self.tokens_processed = 0
            self.last_refresh_total_len = total_len
            return key_states, value_states

        if query_states is not None:
            token_budget = max(self.max_capacity_prompt - self.window_size, 0)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            generated_since_refresh = max(total_len - max(self.last_refresh_total_len, self.max_capacity_prompt), 0)

            def _refresh_prefix():
                current_queries = query_states[:, :, -1, :]
                k_past = key_states[:, :, :-self.window_size, :]
                v_past = value_states[:, :, :-self.window_size, :]
                self.active_pre_rope_query_states = self.expected_query_buffer if self.expected_query_buffer is not None else pre_rope_query_states
                self.active_layer_module = layer_module
                self.active_total_len = total_len
                if self.selection_granularity == 'token':
                    result = self._select_topk_tokens(k_past, v_past, current_queries, query_states)
                elif self.selection_granularity == 'page':
                    result = self._select_topk_pages(k_past, v_past, current_queries, query_states)
                elif self.selection_granularity == 'cluster':
                    result = self._select_topk_clusters(k_past, v_past, current_queries, query_states)
                elif self.selection_granularity == 'clusterattn':
                    result = self._select_clusterattn_density(k_past, v_past, current_queries, query_states)
                else:
                    raise ValueError(f"Unsupported selection granularity: {self.selection_granularity}")
                self.active_pre_rope_query_states = None
                self.active_layer_module = None
                self.active_total_len = 0
                return result

            should_refresh = False
            if self.update_policy == 'static':
                should_refresh = self.compressed_keys is None or self.compressed_values is None
            elif self.update_policy == 'incremental':
                should_refresh = True
            elif self.update_policy == 'periodic':
                should_refresh = (
                    self.compressed_keys is None
                    or self.compressed_values is None
                    or generated_since_refresh >= max(1, self.update_interval)
                )
            else:
                raise ValueError(f"Unsupported update policy: {self.update_policy}")

            if should_refresh:
                k_past_compress, v_past_compress = _refresh_prefix()
                self.compressed_keys = k_past_compress
                self.compressed_values = v_past_compress
                self.last_refresh_total_len = total_len
                self.tokens_processed = max(total_len - self.max_capacity_prompt, 0)
            else:
                if self.compressed_keys is None or self.compressed_values is None:
                    empty = key_states[:, :, :0, :]
                    self.compressed_keys = empty
                    self.compressed_values = value_states[:, :, :0, :]
                if token_budget > 0:
                    self.compressed_keys = self.compressed_keys[:, :, :token_budget, :]
                    self.compressed_values = self.compressed_values[:, :, :token_budget, :]

            key_states = torch.cat([self.compressed_keys, k_cur], dim=2)
            value_states = torch.cat([self.compressed_values, v_cur], dim=2)

        return key_states, value_states


def overwrite_past_key_value(past_key_value, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
    if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
        past_key_value.key_cache[layer_idx] = key_states
        past_key_value.value_cache[layer_idx] = value_states
        return

    if hasattr(past_key_value, "layers"):
        layer = past_key_value.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            layer.keys = key_states
            layer.values = value_states
            return
        if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
            layer.key_cache = key_states
            layer.value_cache = value_states
            return

    raise AttributeError("Unsupported cache structure for overwriting compressed KV states")

def init_clusterkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'n_clusters'):
            self.config.n_clusters = 128
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 64
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'update_policy'):
            self.config.update_policy = 'incremental'
        if not hasattr(self.config, 'update_interval'):
            self.config.update_interval = 100
        if not hasattr(self.config, 'page_size'):
            self.config.page_size = 16
        if not hasattr(self.config, 'ranking_backend'):
            self.config.ranking_backend = 'quest_bounds'
        if not hasattr(self.config, 'observation_window'):
            self.config.observation_window = 32
        if not hasattr(self.config, 'selection_granularity'):
            self.config.selection_granularity = 'cluster'
        if (not hasattr(self.config, 'clustering_backend')) or (getattr(self.config, "clustering_backend") is None):
            self.config.clustering_backend = 'kmeanspp'
        if not hasattr(self.config, 'num_block'):
            self.config.num_block = 12
        if not hasattr(self.config, 'theta'):
            self.config.theta = 0.0
        if not hasattr(self.config, 'n_future_positions'):
            self.config.n_future_positions = 512
        if not hasattr(self.config, 'n_sink'):
            self.config.n_sink = 4
        if not hasattr(self.config, 'use_covariance'):
            self.config.use_covariance = True
        if not hasattr(self.config, 'use_vnorm'):
            self.config.use_vnorm = True
        if not hasattr(self.config, 'epsilon'):
            self.config.epsilon = 0.0
        if not hasattr(self.config, 'hidden_states_buffer_size'):
            self.config.hidden_states_buffer_size = 128

    self.kv_cluster = ClusterKVCache(
        n_clusters=self.config.n_clusters,
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        update_policy=self.config.update_policy,
        update_interval=self.config.update_interval,
        page_size=self.config.page_size,
        ranking_backend=self.config.ranking_backend,
        observation_window=self.config.observation_window,
        selection_granularity=self.config.selection_granularity,
        clustering_backend=self.config.clustering_backend,
        num_block=self.config.num_block,
        theta=self.config.theta,
        n_future_positions=self.config.n_future_positions,
        n_sink=self.config.n_sink,
        use_covariance=self.config.use_covariance,
        use_vnorm=self.config.use_vnorm,
        epsilon=self.config.epsilon,
        hidden_states_buffer_size=self.config.hidden_states_buffer_size,
    )
