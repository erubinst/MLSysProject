import torch
from typing import Optional

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
                 page_size: int = 16):
        """
        Args:
            n_clusters: Number of clusters for k-means
            window_size: Size of recent window to keep uncompressed
            max_capacity_prompt: Maximum capacity for prompt phase
            update_policy: 'static', 'incremental', or 'periodic'
            update_interval: For periodic updates, update every N tokens
            page_size: Number of KV tokens grouped into one retrieval page
        """
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.update_policy = update_policy
        self.update_interval = update_interval
        self.page_size = page_size

        self.kmeans = None
        self.cluster_assignments = None
        self.compressed_keys = None
        self.compressed_values = None
        self.tokens_processed = 0

    def reset(self):
        """Reset the cache"""
        self.kmeans = None
        self.cluster_assignments = None
        self.compressed_keys = None
        self.compressed_values = None
        self.tokens_processed = 0

    def _score_pages(self, prefix_keys: torch.Tensor, current_query: torch.Tensor):
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

        pad_len = (self.page_size - (prefix_len % self.page_size)) % self.page_size
        if pad_len:
            prefix_keys = torch.cat([prefix_keys, prefix_keys.new_zeros(bsz, num_heads, pad_len, head_dim)], dim=2)

        num_pages = prefix_keys.shape[2] // self.page_size
        page_keys = prefix_keys.reshape(bsz, num_heads, num_pages, self.page_size, head_dim)
        page_min = page_keys.amin(dim=3)
        page_max = page_keys.amax(dim=3)

        query = current_query.unsqueeze(2)
        page_scores = torch.maximum(query * page_min, query * page_max).sum(dim=-1)
        return page_scores, page_keys, prefix_len

    def _select_topk_pages(self, prefix_keys: torch.Tensor, prefix_values: torch.Tensor, current_query: torch.Tensor):
        """
        Select top scoring pages from the prefix and preserve chronological order.
        """
        page_scores, page_keys, prefix_len = self._score_pages(prefix_keys, current_query)
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
                  num_key_value_groups: int = 1):
        """
        Update KV cache using page-level query-aware retrieval.
        """
        bsz, num_heads, q_len, head_dim = query_states.shape

        # During prompt phase, keep all tokens if below capacity
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        if query_states is not None:
            current_queries = query_states[:, :, -1, :]
            k_past = key_states[:, :, :-self.window_size, :]
            v_past = value_states[:, :, :-self.window_size, :]
            k_past_compress, v_past_compress = self._select_topk_pages(k_past, v_past, current_queries)

            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]

            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)

        return key_states, value_states

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

    self.kv_cluster = ClusterKVCache(
        n_clusters=self.config.n_clusters,
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        update_policy=self.config.update_policy,
        update_interval=self.config.update_interval,
        page_size=self.config.page_size
    )
