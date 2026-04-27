#!/usr/bin/env python3

"""
Test script for ClusterKV implementation
"""

import torch
from snapkv.monkeypatch.clusterkv_utils import OnlineKMeans, ClusterKVCache

def test_online_kmeans():
    """Test the OnlineKMeans class"""
    print("Testing OnlineKMeans...")

    # Create test data
    n_samples = 1000
    dim = 128
    n_clusters = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.randn(n_samples, dim, device=device)

    # Initialize k-means
    kmeans = OnlineKMeans(n_clusters, dim, device)

    # Test initialization
    kmeans.initialize(data)
    assert kmeans.initialized
    assert kmeans.centroids.shape == (n_clusters, dim)
    print("✓ Initialization successful")

    # Test update
    new_data = torch.randn(100, dim, device=device)
    kmeans.update(new_data)
    print("✓ Update successful")

    # Test quantization
    quantized, centroids_min, scale = kmeans.quantize_centroids_int8()
    assert quantized.dtype == torch.int8
    assert quantized.shape == (n_clusters, dim)
    print("✓ Quantization successful")

    # Test top-k retrieval
    query = torch.randn(1, dim, device=device)
    top_k = kmeans.get_top_k_clusters(query, k=10)
    assert top_k.shape == (10,)
    print("✓ Top-K retrieval successful")

def test_cluster_kv_cache():
    """Test the ClusterKVCache class"""
    print("\nTesting ClusterKVCache...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test tensors (batch_size=1, num_heads=8, seq_len=100, head_dim=128)
    bsz, num_heads, seq_len, head_dim = 1, 8, 100, 128
    key_states = torch.randn(bsz, num_heads, seq_len, head_dim, device=device)
    query_states = torch.randn(bsz, num_heads, seq_len, head_dim, device=device)
    value_states = torch.randn(bsz, num_heads, seq_len, head_dim, device=device)

    # Initialize cache
    cache = ClusterKVCache(n_clusters=32, window_size=64, max_capacity_prompt=4096, page_size=16)

    # Test update
    key_out, value_out = cache.update_kv(key_states, query_states, value_states)
    assert key_out.shape == key_states.shape
    assert value_out.shape == value_states.shape
    print("✓ Cache update successful")

def test_page_level_retrieval():
    """Quest-style page scoring should retrieve the most relevant page."""
    print("\nTesting page-level retrieval...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    cache = ClusterKVCache(
        n_clusters=4,
        window_size=2,
        max_capacity_prompt=4,
        page_size=2,
    )

    # Prefix has two pages:
    # page 0 => weak / negative match for positive query
    # page 1 => strong positive match and should be selected
    key_states = torch.tensor(
        [[[
            [-2.0, 0.0],
            [-1.0, 0.0],
            [5.0, 0.0],
            [4.0, 0.0],
            [0.5, 0.0],
            [0.25, 0.0],
        ]]],
        device=device,
        dtype=dtype,
    )
    value_states = key_states.clone()
    query_states = torch.tensor(
        [[[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]]],
        device=device,
        dtype=dtype,
    )

    key_out, value_out = cache.update_kv(key_states, query_states, value_states)

    expected_prefix = key_states[:, :, 2:4, :]
    expected_window = key_states[:, :, -2:, :]

    assert key_out.shape == (1, 1, 4, 2)
    assert value_out.shape == (1, 1, 4, 2)
    assert torch.allclose(key_out[:, :, :2, :], expected_prefix)
    assert torch.allclose(key_out[:, :, 2:, :], expected_window)
    assert torch.allclose(value_out, key_out)
    print("✓ Page-level retrieval selected the highest scoring page")

def test_cluster_level_retrieval():
    """Cluster-level retrieval should select semantically grouped relevant tokens."""
    print("\nTesting cluster-level retrieval...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    torch.manual_seed(0)

    cache = ClusterKVCache(
        n_clusters=2,
        window_size=2,
        max_capacity_prompt=4,
        ranking_backend="quest_bounds",
        selection_granularity="cluster",
    )

    # Prefix tokens form two clear semantic groups:
    # cluster A ~ strongly negative on dim 0
    # cluster B ~ strongly positive on dim 0
    # With a positive query, cluster B should be recalled.
    key_states = torch.tensor(
        [[[
            [-5.0, 0.0],
            [-4.5, 0.0],
            [4.5, 0.0],
            [5.0, 0.0],
            [0.2, 0.0],
            [0.1, 0.0],
        ]]],
        device=device,
        dtype=dtype,
    )
    value_states = key_states.clone()
    query_states = torch.tensor(
        [[[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]]],
        device=device,
        dtype=dtype,
    )

    key_out, value_out = cache.update_kv(key_states, query_states, value_states)

    expected_prefix = key_states[:, :, 2:4, :]
    expected_window = key_states[:, :, -2:, :]

    assert key_out.shape == (1, 1, 4, 2)
    assert value_out.shape == (1, 1, 4, 2)
    assert torch.allclose(key_out[:, :, :2, :], expected_prefix)
    assert torch.allclose(key_out[:, :, 2:, :], expected_window)
    assert torch.allclose(value_out, key_out)
    print("✓ Cluster-level retrieval selected the relevant k-means cluster")

def test_cluster_backends_smoke():
    """All configured cluster backends should execute through the cluster selector."""
    print("\nTesting cluster backend smoke cases...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    torch.manual_seed(0)

    key_states = torch.tensor(
        [[[
            [-5.0, 0.0],
            [-4.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [0.2, 0.0],
            [0.1, 0.0],
        ]]],
        device=device,
        dtype=dtype,
    )
    value_states = key_states.clone()
    query_states = torch.tensor(
        [[[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]]],
        device=device,
        dtype=dtype,
    )

    for backend in ["kmeans", "kmeanspp", "spherical_kmeans"]:
        cache = ClusterKVCache(
            n_clusters=2,
            window_size=2,
            max_capacity_prompt=4,
            ranking_backend="quest_bounds",
            selection_granularity="cluster",
            clustering_backend=backend,
        )
        key_out, value_out = cache.update_kv(key_states, query_states, value_states)
        assert key_out.shape == (1, 1, 4, 2)
        assert value_out.shape == (1, 1, 4, 2)
    print("✓ Cluster backends executed successfully")

def test_clusterattn_backends_smoke():
    """ClusterAttn should accept all supported token scoring backends."""
    print("\nTesting ClusterAttn backend smoke cases...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    torch.manual_seed(0)

    key_states = torch.tensor(
        [[[
            [-3.0, 0.0],
            [-2.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ]]],
        device=device,
        dtype=dtype,
    )
    value_states = key_states.clone()
    query_states = torch.tensor(
        [[[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]]],
        device=device,
        dtype=dtype,
    )

    for backend in [
        "quest_bounds",
        "snapkv_prefill",
        "h2o_accum",
        "reconstruction_error",
        "expected_attention",
        "random",
    ]:
        cache = ClusterKVCache(
            n_clusters=2,
            window_size=2,
            max_capacity_prompt=4,
            ranking_backend=backend,
            selection_granularity="clusterattn",
            num_block=2,
            theta=0.0,
        )
        key_out, value_out = cache.update_kv(key_states, query_states, value_states)
        assert key_out.shape == (1, 1, 4, 2)
        assert value_out.shape == (1, 1, 4, 2)
    print("✓ ClusterAttn backends executed successfully")

if __name__ == "__main__":
    test_online_kmeans()
    test_cluster_kv_cache()
    test_page_level_retrieval()
    test_cluster_level_retrieval()
    test_cluster_backends_smoke()
    test_clusterattn_backends_smoke()
    print("\nAll tests passed!")
