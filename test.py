import modal
from datetime import datetime

app = modal.App("clusterkv-eval")
volume = modal.Volume.from_name("clusterkv-models", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install("wheel", "packaging")
    .pip_install(
        "torch==2.4.0",
        "accelerate",
        "datasets==2.16.0",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("transformers==4.37.0", "huggingface-hub==0.36.2")
    .pip_install("fschat==0.2.36", "sentencepiece", "protobuf")
    .pip_install("xgboost", "scikit-learn")
    .pip_install(
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu121torch2.4-cp311-cp311-linux_x86_64.whl"
    )
    .pip_install("jieba", "rouge", "fuzzywuzzy", "python-Levenshtein")
    .add_local_dir(".", remote_path="/app", copy=True, ignore=[".venv"])
    .run_commands("cd /app && pip install -e . --no-deps")
)

# ============================================================
# Datasets and methods config
# ============================================================

DATASETS = ["qasper", "hotpotqa", "gov_report", "lcc"]
VALIDATION_DATASET = "hotpotqa"
VALIDATION_SAMPLE_OFFSET = 0
XGB_ROUTER_SAMPLE_LIMIT = 100

STATIC_METHODS = [
    "baseline",
    "snapkv_static",
    "quest_static",
    "clusterattn_static",
    "clusterattn_quest_bounds_static",
    "clusterattn_snapkv_static",
    "clusterattn_h2o_static",
    "clusterattn_recon_static",
    "clusterattn_expected_attention_static",
    "clusterattn_random_static",
    "pagekv_quest_bounds_static",
    "pagekv_snapkv_static",
    "pagekv_h2o_static",
    "pagekv_recon_static",
    "pagekv_expected_attention_static",
    "pagekv_random_static",
    "tokenkv_quest_bounds_static",
    "tokenkv_snapkv_static",
    "tokenkv_h2o_static",
    "tokenkv_recon_static",
    "tokenkv_expected_attention_static",
    "tokenkv_random_static",
]

DYNAMIC_METHODS = [
    "clusterattn_quest_bounds_dynamic",
    "clusterattn_h2o_dynamic",
    "clusterattn_expected_attention_dynamic",
    "clusterattn_random_dynamic",
    "pagekv_quest_bounds_dynamic",
    "pagekv_h2o_dynamic",
    "pagekv_expected_attention_dynamic",
    "pagekv_random_dynamic",
    "tokenkv_quest_bounds_dynamic",
    "tokenkv_h2o_dynamic",
    "tokenkv_expected_attention_dynamic",
    "tokenkv_random_dynamic",
]

XGB_ROUTER_CANDIDATE_METHODS = [
    "tokenkv_quest_bounds_dynamic100",
    "clusterattn_recon_static",
    "clusterattn_quest_bounds_static",
    "pagekv_quest_bounds_static",
    "tokenkv_quest_bounds_static",
    "tokenkv_h2o_dynamic",
]

# Default grid for main_topk_ablation (override with --base-methods-csv / --k-rets-csv).
TOPK_ABLATION_DEFAULT_BASE_METHODS = [
    "tokenkv_snapkv_static",
    "clusterattn_snapkv_static",
    "clusterattn_expected_attention_static",
]
TOPK_ABLATION_DEFAULT_K_RETS = [512, 1024, 2048, 3072, 4032]


def _build_run_tag(version: str = "1", run_tag: str = "") -> str:
    if run_tag:
        return run_tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = str(version).strip()
    if not version.startswith("v"):
        version = f"v{version}"
    return f"{version}_{timestamp}"


def _predictions_dir(run_tag: str, method: str) -> str:
    return f"/models/runs/{run_tag}/predictions/{method}"


def _results_dir(run_tag: str) -> str:
    return f"/models/runs/{run_tag}/results"


def _validations_dir(run_tag: str) -> str:
    return f"/models/runs/{run_tag}/validations"


METHODS = {
    "baseline": {
        "script": "pred_snap.py",
        "extra_args": [],
        "model_name": "mistral-7B-instruct-v0.2",
    },
    "baseline_clusterpath_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "baseline_clusterpath_fullkv.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2baseline_clusterpath_fullkv",
    },
    "snapkv_static": {
        "script": "pred_snap.py",
        "extra_args": ["--compress_args_path", "ablation_c4096_w32_k7_maxpool.json"],
        "model_name": "mistral-7B-instruct-v0.2ablation_c4096_w32_k7_maxpool",
    },
    "quest_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "quest",
            "--compress_args_path", "quest_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2quest_c4096_w64_p16",
    },
    "clusterattn_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_clusterattn_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_static_c4096_w64_p16",
    },
    "clusterattn_quest_bounds_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_quest_bounds_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_quest_bounds_static_c4096_w64_p16",
    },
    "clusterattn_snapkv_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_snapkv_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_snapkv_static_c4096_w64_p16",
    },
    "clusterattn_h2o_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_h2o_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_h2o_static_c4096_w64_p16",
    },
    "clusterattn_recon_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_recon_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_recon_static_c4096_w64_p16",
    },
    "clusterattn_expected_attention_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_expected_attention_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_expected_attention_static_c4096_w64_p16",
    },
    "clusterattn_random_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_random_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_random_static_c4096_w64_p16",
    },
    "clusterattn_quest_bounds_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_quest_bounds_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_quest_bounds_dynamic_c4096_w64_p16",
    },
    "clusterattn_h2o_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_h2o_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_h2o_dynamic_c4096_w64_p16",
    },
    "clusterattn_expected_attention_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_expected_attention_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_expected_attention_dynamic_c4096_w64_p16",
    },
    "clusterattn_random_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_random_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_random_dynamic_c4096_w64_p16",
    },
    "clusterkv_quest_bounds_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_c4096_w64_p16",
    },
    "clusterkv_quest_bounds_kmeans_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_kmeans_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_kmeans_c4096_w64_p16",
    },
    "clusterkv_quest_bounds_spherical_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_spherical_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_spherical_c4096_w64_p16",
    },
    "clusterkv_snapkv_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_snapkv_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_snapkv_c4096_w64_p16",
    },
    "clusterkv_snapkv_kmeans_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_snapkv_kmeans_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_snapkv_kmeans_c4096_w64_p16",
    },
    "clusterkv_snapkv_spherical_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_snapkv_spherical_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_snapkv_spherical_c4096_w64_p16",
    },
    "clusterkv_h2o_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_h2o_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_h2o_c4096_w64_p16",
    },
    "clusterkv_h2o_kmeans_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_h2o_kmeans_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_h2o_kmeans_c4096_w64_p16",
    },
    "clusterkv_h2o_spherical_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_h2o_spherical_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_h2o_spherical_c4096_w64_p16",
    },
    "clusterkv_recon_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_recon_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_recon_c4096_w64_p16",
    },
    "clusterkv_recon_kmeans_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_recon_kmeans_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_recon_kmeans_c4096_w64_p16",
    },
    "clusterkv_recon_spherical_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_recon_spherical_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_recon_spherical_c4096_w64_p16",
    },
    "clusterkv_expected_attention_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_expected_attention_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_expected_attention_c4096_w64_p16",
    },
    "clusterkv_expected_attention_kmeans_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_expected_attention_kmeans_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_expected_attention_kmeans_c4096_w64_p16",
    },
    "clusterkv_expected_attention_spherical_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_expected_attention_spherical_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_expected_attention_spherical_c4096_w64_p16",
    },
    "clusterkv_random_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_random_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_random_c4096_w64_p16",
    },
    "clusterkv_random_kmeans_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_random_kmeans_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_random_kmeans_c4096_w64_p16",
    },
    "clusterkv_random_spherical_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterkv_random_spherical_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterkv_random_spherical_c4096_w64_p16",
    },
    "pagekv_quest_bounds_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_quest_bounds_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_quest_bounds_c4096_w64_p16",
    },
    "pagekv_snapkv_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_snapkv_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_snapkv_c4096_w64_p16",
    },
    "pagekv_h2o_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_h2o_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_h2o_c4096_w64_p16",
    },
    "pagekv_recon_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_recon_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_recon_c4096_w64_p16",
    },
    "pagekv_expected_attention_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_expected_attention_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_expected_attention_c4096_w64_p16",
    },
    "pagekv_random_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_random_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_random_c4096_w64_p16",
    },
    "pagekv_quest_bounds_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_quest_bounds_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_quest_bounds_dynamic_c4096_w64_p16",
    },
    "pagekv_h2o_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_h2o_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_h2o_dynamic_c4096_w64_p16",
    },
    "pagekv_expected_attention_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_expected_attention_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_expected_attention_dynamic_c4096_w64_p16",
    },
    "pagekv_random_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_random_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_random_dynamic_c4096_w64_p16",
    },
    "tokenkv_quest_bounds_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_quest_bounds_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_quest_bounds_c4096_w64",
    },
    "tokenkv_snapkv_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_snapkv_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_snapkv_c4096_w64",
    },
    "tokenkv_h2o_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_h2o_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_h2o_c4096_w64",
    },
    "tokenkv_recon_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_recon_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_recon_c4096_w64",
    },
    "tokenkv_expected_attention_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_expected_attention_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_expected_attention_c4096_w64",
    },
    "tokenkv_random_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_random_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_random_c4096_w64",
    },
    "tokenkv_quest_bounds_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_quest_bounds_dynamic_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_quest_bounds_dynamic_c4096_w64",
    },
    "tokenkv_quest_bounds_dynamic100": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_quest_bounds_dynamic100_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_quest_bounds_dynamic100_c4096_w64",
    },
    "tokenkv_h2o_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_h2o_dynamic_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_h2o_dynamic_c4096_w64",
    },
    "tokenkv_expected_attention_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_expected_attention_dynamic_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_expected_attention_dynamic_c4096_w64",
    },
    "tokenkv_random_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_random_dynamic_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_random_dynamic_c4096_w64",
    },
    "h2o_static": {
        "script": "pred_h2o.py",
        "extra_args": ["--max_capacity_prompt", "2048", "--window_size", "32"],
        "model_name": "mistral-7B-instruct-v0.2h2o_static_budget2048",
    },
    "heuristic_routing": {
        "script": "pred_snap.py",
        "extra_args": ["--method", "heuristic_routing"],
        "model_name": "mistral-7B-instruct-v0.2heuristic_routing",
    },
    "xgb_routing": {
        "script": "pred_snap.py",
        "extra_args": ["--method", "xgb_routing"],
        "model_name": "mistral-7B-instruct-v0.2xgb_routing",
    },
}

# Top-k ablation: synthetic method id is "{base_method}__kret{k_ret}" where
# k_ret = max_capacity_prompt - window_size (see clusterkv_utils token_budget).
TOPK_METHOD_DELIM = "__kret"


def _parse_topk_ablation_method(method: str) -> tuple[str, int] | None:
    if TOPK_METHOD_DELIM not in method:
        return None
    base, sep, k_part = method.rpartition(TOPK_METHOD_DELIM)
    if not base or sep != TOPK_METHOD_DELIM or not k_part.isdigit():
        return None
    return base, int(k_part)


def _materialize_topk_method_config(base_method: str, k_ret: int) -> dict:
    import json
    import os

    if base_method not in METHODS:
        raise KeyError(f"Unknown base method {base_method!r} for top-k ablation")
    base_cfg = METHODS[base_method]
    extra = list(base_cfg["extra_args"])
    rel_template = None
    for i, a in enumerate(extra):
        if a == "--compress_args_path" and i + 1 < len(extra):
            rel_template = extra[i + 1]
            break
    if rel_template is None:
        raise ValueError(
            f"Method {base_method!r} has no --compress_args_path; top-k ablation only applies "
            "to clusterkv/quest configs under experiments/LongBench/config."
        )
    config_root = "/app/experiments/LongBench/config"
    src_path = os.path.join(config_root, rel_template)
    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)
    if "window_size" not in data or "max_capacity_prompt" not in data:
        raise ValueError(
            f"Template {rel_template!r} must contain window_size and max_capacity_prompt for top-k ablation."
        )
    window_size = int(data["window_size"])
    data["max_capacity_prompt"] = window_size + int(k_ret)
    out_rel = f"_topk_ablation/{base_method}_kret{k_ret}.json"
    out_path = os.path.join(config_root, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    for i, a in enumerate(extra):
        if a == "--compress_args_path" and i + 1 < len(extra):
            extra[i + 1] = out_rel
            break
    model_name = base_cfg["model_name"] + f"_kret{k_ret}"
    return {
        "script": base_cfg["script"],
        "extra_args": extra,
        "model_name": model_name,
    }


def _is_registered_method(method: str) -> bool:
    if method in METHODS:
        return True
    parsed = _parse_topk_ablation_method(method)
    return parsed is not None and parsed[0] in METHODS


def _resolve_method_config(method: str, dataset: str | None = None):
    parsed = _parse_topk_ablation_method(method)
    if parsed is not None:
        base_method, k_ret = parsed
        return _materialize_topk_method_config(base_method, k_ret)
    return METHODS[method]


def _execute_inference_run(
    method: str,
    cfg: dict,
    dataset: str,
    run_tag: str,
    limit: int | None = None,
    sample_offset: int = 0,
    router_run_tag: str = ""):
    import json
    import os
    import re
    import shutil
    import subprocess

    cmd = [
        "python",
        cfg["script"],
        "--model",
        "mistral-7B-instruct-v0.2",
        "--dataset",
        dataset,
        "--write_model_name",
        cfg["model_name"],
    ] + cfg["extra_args"]
    if method == "xgb_routing":
        if not router_run_tag:
            raise ValueError("xgb_routing requires router_run_tag")
        cmd += ["--xgb_router_dir", f"/models/runs/{router_run_tag}/results/router_data"]
    if limit is not None:
        cmd += ["--limit", str(limit), "--sample_offset", str(sample_offset)]

    print(f"\n{'='*50}")
    print(f"Running {method} on {dataset}")
    if limit is not None:
        print(f"Sample slice: offset={sample_offset}, limit={limit}")
    print(f"Run tag: {run_tag}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")

    result = subprocess.run(
        cmd, cwd="/app/experiments/LongBench", capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Inference failed for {method} / {dataset} with exit code {result.returncode}"
        )

    pred_dir = _predictions_dir(run_tag, method)
    os.makedirs(pred_dir, exist_ok=True)
    for d in ["pred", "pred_e"]:
        src = f"/app/experiments/LongBench/{d}"
        if os.path.exists(src):
            shutil.copytree(src, f"{pred_dir}/{d}", dirs_exist_ok=True)
            print(f"Saved {src} to {pred_dir}/{d}")

    peak_match = re.search(r"Peak GPU memory allocated:\s+([0-9.]+)\s+MB\s+\(([0-9.]+)\s+GB\)", result.stdout)
    kv_match = re.search(r"Memory for KV cache \(approx\):\s+([0-9.]+)\s+MB", result.stdout)
    total_latency_match = re.search(r"Total latency:\s+([0-9.]+)\s+s", result.stdout)
    avg_prefill_latency_match = re.search(r"Average prefill latency:\s+([0-9.]+)\s+s", result.stdout)
    avg_decode_latency_match = re.search(r"Average decode latency:\s+([0-9.]+)\s+s", result.stdout)
    max_prefill_latency_match = re.search(r"Max prefill latency:\s+([0-9.]+)\s+s", result.stdout)
    max_decode_latency_match = re.search(r"Max decode latency:\s+([0-9.]+)\s+s", result.stdout)
    avg_latency_match = re.search(r"Average latency / example:\s+([0-9.]+)\s+s", result.stdout)
    max_latency_match = re.search(r"Max latency / example:\s+([0-9.]+)\s+s", result.stdout)
    throughput_match = re.search(r"Generation throughput:\s+([0-9.]+)\s+tok/s", result.stdout)
    profiled_flops_match = re.search(r"Profiled FLOPs \(1st example\):\s+([0-9.]+)", result.stdout)
    profiled_tflops_match = re.search(r"Profiled TFLOPs \(1st example\):\s+([0-9.]+)", result.stdout)
    profiled_tflops_per_s_match = re.search(r"Profiled TFLOPs/s \(1st example\):\s+([0-9.]+)", result.stdout)

    required_metric_matches = {
        "peak GPU memory": peak_match,
        "KV cache memory": kv_match,
        "total latency": total_latency_match,
        "average latency": avg_latency_match,
        "throughput": throughput_match,
        "profiled FLOPs": profiled_flops_match,
        "profiled TFLOPs": profiled_tflops_match,
        "profiled TFLOPs/s": profiled_tflops_per_s_match,
    }
    missing_metric_names = [
        name for name, match in required_metric_matches.items()
        if match is None
    ]
    if missing_metric_names:
        raise RuntimeError(
            f"Inference for {method} / {dataset} completed without required metric logs: "
            + ", ".join(missing_metric_names)
        )

    peak_gb = float(peak_match.group(2)) if peak_match else None
    kv_cache_mb = float(kv_match.group(1)) if kv_match else None
    total_latency_s = float(total_latency_match.group(1)) if total_latency_match else None
    avg_prefill_latency_s = float(avg_prefill_latency_match.group(1)) if avg_prefill_latency_match else None
    avg_decode_latency_s = float(avg_decode_latency_match.group(1)) if avg_decode_latency_match else None
    max_prefill_latency_s = float(max_prefill_latency_match.group(1)) if max_prefill_latency_match else None
    max_decode_latency_s = float(max_decode_latency_match.group(1)) if max_decode_latency_match else None
    avg_latency_s = float(avg_latency_match.group(1)) if avg_latency_match else None
    max_latency_s = float(max_latency_match.group(1)) if max_latency_match else None
    tokens_per_second = float(throughput_match.group(1)) if throughput_match else None
    profiled_flops = float(profiled_flops_match.group(1)) if profiled_flops_match else None
    profiled_tflops = float(profiled_tflops_match.group(1)) if profiled_tflops_match else None
    profiled_tflops_per_s = float(profiled_tflops_per_s_match.group(1)) if profiled_tflops_per_s_match else None

    os.makedirs(_results_dir(run_tag), exist_ok=True)
    memory_result_path = f"{_results_dir(run_tag)}/{method}_{dataset}_memory.json"
    with open(memory_result_path, "w") as f:
        json.dump(
            {
                "run_tag": run_tag,
                "method": method,
                "dataset": dataset,
                "peak_gb": peak_gb,
                "kv_cache_mb": kv_cache_mb,
                "total_latency_s": total_latency_s,
                "avg_prefill_latency_s": avg_prefill_latency_s,
                "avg_decode_latency_s": avg_decode_latency_s,
                "max_prefill_latency_s": max_prefill_latency_s,
                "max_decode_latency_s": max_decode_latency_s,
                "avg_latency_s": avg_latency_s,
                "max_latency_s": max_latency_s,
                "tokens_per_second": tokens_per_second,
                "profiled_flops": profiled_flops,
                "profiled_tflops": profiled_tflops,
                "profiled_tflops_per_s": profiled_tflops_per_s,
            },
            f,
        )
    print(f"Saved memory stats to {memory_result_path}")

    metadata_path = f"{_results_dir(run_tag)}/{method}_{dataset}_inference.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "run_tag": run_tag,
                "method": method,
                "dataset": dataset,
                "script": cfg["script"],
                "extra_args": cfg["extra_args"],
                "routed_method": cfg.get("routed_method"),
                "limit": limit,
                "sample_offset": sample_offset,
                "router_run_tag": router_run_tag,
                "returncode": result.returncode,
            },
            f,
        )
    print(f"Saved inference metadata to {metadata_path}")


# ============================================================
# Inference
# ============================================================

@app.function(
    gpu="A100",
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200
)
def run_inference(method: str, dataset: str, run_tag: str, limit: int | None = None, sample_offset: int = 0):
    cfg = _resolve_method_config(method, dataset)
    _execute_inference_run(method, cfg, dataset, run_tag, limit, sample_offset)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200
)
def run_inference_a100_80gb(method: str, dataset: str, run_tag: str, limit: int | None = None, sample_offset: int = 0):
    cfg = _resolve_method_config(method, dataset)
    _execute_inference_run(method, cfg, dataset, run_tag, limit, sample_offset)


@app.function(
    gpu="A100",
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200
)
def run_inference_topk(
    base_method: str,
    k_ret: int,
    dataset: str,
    run_tag: str,
    limit: int | None = None,
    sample_offset: int = 0,
):
    """Run one (base_method, k_ret) cell; method id is base_method + '__kret' + str(k_ret)."""
    method = f"{base_method}{TOPK_METHOD_DELIM}{k_ret}"
    cfg = _resolve_method_config(method, dataset)
    _execute_inference_run(method, cfg, dataset, run_tag, limit, sample_offset)


@app.function(
    gpu="A100",
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=2400
)
def run_validation(method: str, dataset: str = VALIDATION_DATASET, sample_offset: int = VALIDATION_SAMPLE_OFFSET, run_tag: str = "unversioned", router_run_tag: str = ""):
    import json, os, re, shutil, subprocess

    cfg = _resolve_method_config(method, dataset)
    cmd = [
        "python", cfg["script"],
        "--model", "mistral-7B-instruct-v0.2",
        "--dataset", dataset,
        "--limit", "1",
        "--sample_offset", str(sample_offset),
        "--write_model_name", cfg["model_name"],
    ] + cfg["extra_args"]
    if method == "xgb_routing":
        if not router_run_tag:
            raise ValueError("xgb_routing requires router_run_tag")
        cmd += ["--xgb_router_dir", f"/models/runs/{router_run_tag}/results/router_data"]

    print(f"\n{'='*50}")
    print(f"Validating {method} on {dataset} sample {sample_offset}")
    print(f"Run tag: {run_tag}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")

    result = subprocess.run(
        cmd, cwd="/app/experiments/LongBench",
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    pred_dir = f"{_predictions_dir(run_tag, method)}_validation"
    os.makedirs(pred_dir, exist_ok=True)
    for d in ["pred", "pred_e"]:
        src = f"/app/experiments/LongBench/{d}"
        if os.path.exists(src):
            shutil.copytree(src, f"{pred_dir}/{d}", dirs_exist_ok=True)

    output_model_name = cfg["model_name"]
    pred_file = f"/app/experiments/LongBench/pred_e/{output_model_name}/{dataset}.jsonl"
    record = None
    if os.path.exists(pred_file):
        with open(pred_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if line:
                record = json.loads(line)

    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return text

    validation = {
        "run_tag": run_tag,
        "method": method,
        "dataset": dataset,
        "sample_offset": sample_offset,
        "command": cmd,
        "routed_method": cfg.get("routed_method"),
        "router_run_tag": router_run_tag,
        "ran_successfully": result.returncode == 0,
        "pred_file_found": record is not None,
    }

    if record is not None:
        pred = record.get("pred", "")
        answers = record.get("answers", [])
        pred_norm = _normalize(pred)
        answers_norm = [_normalize(answer) for answer in answers if isinstance(answer, str)]
        validation.update(
            {
                "pred": pred,
                "answers": answers,
                "nonempty_pred": bool(pred.strip()),
                "exact_any": pred_norm in answers_norm if pred_norm else False,
                "contains_any_answer": any(answer and answer in pred_norm for answer in answers_norm) if pred_norm else False,
            }
        )

    os.makedirs(_validations_dir(run_tag), exist_ok=True)
    validation_path = f"{_validations_dir(run_tag)}/{method}_{dataset}_sample{sample_offset}.json"
    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)
    print(f"Saved validation artifact to {validation_path}")

    return validation


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800
)
def submit_inference_batch(methods: list[str], run_tag: str):
    submitted = 0
    for method in methods:
        for dataset in DATASETS:
            print(f"Submitting {method} / {dataset}...")
            run_inference.spawn(method, dataset, run_tag)
            submitted += 1
    print(f"Submitted {submitted} inference jobs for run {run_tag}.")
    return submitted


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800
)
def submit_inference_limited_batch(methods: list[str], run_tag: str, limit: int, sample_offset: int = 0):
    submitted = 0
    for method in methods:
        for dataset in DATASETS:
            print(f"Submitting {method} / {dataset} offset={sample_offset} limit={limit}...")
            run_inference.spawn(method, dataset, run_tag, limit, sample_offset)
            submitted += 1
    print(f"Submitted {submitted} limited inference jobs for run {run_tag}.")
    return submitted


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800
)
def submit_xgb_routing_batch(run_tag: str, router_run_tag: str):
    submitted = 0
    for dataset in DATASETS:
        print(f"Submitting xgb_routing / {dataset} using router {router_run_tag}...")
        run_inference.spawn("xgb_routing", dataset, run_tag, None, 0, router_run_tag)
        submitted += 1
    print(f"Submitted {submitted} xgb_routing inference jobs for run {run_tag}.")
    return submitted


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800
)
def submit_eval_batch(methods: list[str], run_tag: str):
    submitted = 0
    for method in methods:
        for dataset in DATASETS:
            print(f"Evaluating {method} / {dataset}...")
            run_eval.spawn(method, dataset, run_tag)
            submitted += 1
    print(f"Submitted {submitted} eval jobs for run {run_tag}.")
    return submitted


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800
)
def submit_validation_batch(methods: list[str], run_tag: str, dataset: str = VALIDATION_DATASET, sample_offset: int = VALIDATION_SAMPLE_OFFSET):
    submitted = 0
    for method in methods:
        print(f"Validating {method} on {dataset} sample {sample_offset}...")
        run_validation.spawn(method, dataset, sample_offset, run_tag)
        submitted += 1
    print(f"Submitted {submitted} validation jobs for run {run_tag}.")
    return submitted


# ============================================================
# Eval
# ============================================================

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=300
)
def run_eval(method: str, dataset: str, run_tag: str):
    import subprocess, shutil, os, json

    cfg = _resolve_method_config(method, dataset)
    model_name = cfg["model_name"]
    pred_dir = _predictions_dir(run_tag, method)

    # Copy predictions into expected location
    for d in ["pred", "pred_e"]:
        src = f"{pred_dir}/{d}"
        dst = f"/app/experiments/LongBench/{d}"
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    os.makedirs(
        f"/app/experiments/LongBench/H2O/results/{model_name}",
        exist_ok=True
    )

    result = subprocess.run(
        ["python", "eval.py", "--model", model_name],
        cwd="/app/experiments/LongBench",
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    result_path = f"/app/experiments/LongBench/H2O/results/{model_name}/result.json"
    score = None
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = json.load(f)
            score = results.get(dataset)
            print(f"\n=== {method.upper()} | {dataset} | F1: {score} ===\n")

        # Persist result to volume
        os.makedirs(_results_dir(run_tag), exist_ok=True)
        volume_result_path = f"{_results_dir(run_tag)}/{method}_{dataset}.json"
        with open(volume_result_path, "w") as f:
            json.dump({"run_tag": run_tag, "method": method, "dataset": dataset, "score": score}, f)

    return score


# ============================================================
# CSV summary
# ============================================================

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=120
)
def generate_csv(run_tag: str):
    import os, json, csv, io

    results_dir = _results_dir(run_tag)
    if not os.path.exists(results_dir):
        print(f"No results found yet for run {run_tag}.")
        return

    # Collect all result files
    rows = {}  # {method: {dataset: score}}
    memory = {}

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        method = data["method"]
        dataset = data["dataset"]

        if "score" in data:
            score = data["score"]
            if method not in rows:
                rows[method] = {}
            rows[method][dataset] = score

        if "peak_gb" in data or "kv_cache_mb" in data:
            if method not in memory:
                memory[method] = {}
            memory[method][dataset] = {
                "peak_gb": data.get("peak_gb"),
                "kv_cache_mb": data.get("kv_cache_mb"),
                "total_latency_s": data.get("total_latency_s"),
                "avg_prefill_latency_s": data.get("avg_prefill_latency_s"),
                "avg_decode_latency_s": data.get("avg_decode_latency_s"),
                "max_prefill_latency_s": data.get("max_prefill_latency_s"),
                "max_decode_latency_s": data.get("max_decode_latency_s"),
                "avg_latency_s": data.get("avg_latency_s"),
                "max_latency_s": data.get("max_latency_s"),
                "tokens_per_second": data.get("tokens_per_second"),
                "profiled_flops": data.get("profiled_flops"),
                "profiled_tflops": data.get("profiled_tflops"),
                "profiled_tflops_per_s": data.get("profiled_tflops_per_s"),
            }

    if not rows:
        print(f"No scored results found yet for run {run_tag}.")
        return

    # Build CSV
    all_datasets = sorted({d for m in rows.values() for d in m})
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(
        ["Method"] + all_datasets + [
            "Average",
            "Peak GPU (GB)",
            "KV Cache (MB)",
            "Avg Latency (s)",
            "Avg Prefill Latency (s)",
            "Avg Decode Latency (s)",
            "Max Prefill Latency (s)",
            "Max Decode Latency (s)",
            "Throughput (tok/s)",
            "Profiled TFLOPs",
            "Profiled TFLOPs/s",
        ]
    )

    for method in [
        "baseline",
        "baseline_clusterpath_static",
        "heuristic_routing",
        "xgb_routing",
        "snapkv_static",
        "quest_static",
        "clusterattn_static",
        "clusterattn_quest_bounds_static",
        "clusterattn_snapkv_static",
        "clusterattn_h2o_static",
        "clusterattn_recon_static",
        "clusterattn_expected_attention_static",
        "clusterattn_random_static",
        "clusterattn_quest_bounds_dynamic",
        "clusterattn_h2o_dynamic",
        "clusterattn_expected_attention_dynamic",
        "clusterattn_random_dynamic",
        "clusterkv_quest_bounds_static",
        "clusterkv_quest_bounds_kmeans_static",
        "clusterkv_quest_bounds_spherical_static",
        "clusterkv_snapkv_static",
        "clusterkv_snapkv_kmeans_static",
        "clusterkv_snapkv_spherical_static",
        "clusterkv_h2o_static",
        "clusterkv_h2o_kmeans_static",
        "clusterkv_h2o_spherical_static",
        "clusterkv_recon_static",
        "clusterkv_recon_kmeans_static",
        "clusterkv_recon_spherical_static",
        "clusterkv_expected_attention_static",
        "clusterkv_expected_attention_kmeans_static",
        "clusterkv_expected_attention_spherical_static",
        "clusterkv_random_static",
        "clusterkv_random_kmeans_static",
        "clusterkv_random_spherical_static",
        "pagekv_quest_bounds_static",
        "pagekv_snapkv_static",
        "pagekv_h2o_static",
        "pagekv_recon_static",
        "pagekv_expected_attention_static",
        "pagekv_random_static",
        "pagekv_quest_bounds_dynamic",
        "pagekv_h2o_dynamic",
        "pagekv_expected_attention_dynamic",
        "pagekv_random_dynamic",
        "tokenkv_quest_bounds_static",
        "tokenkv_snapkv_static",
        "tokenkv_h2o_static",
        "tokenkv_recon_static",
        "tokenkv_expected_attention_static",
        "tokenkv_random_static",
        "tokenkv_quest_bounds_dynamic",
        "tokenkv_quest_bounds_dynamic100",
        "tokenkv_h2o_dynamic",
        "tokenkv_expected_attention_dynamic",
        "tokenkv_random_dynamic",
        "h2o_static",
        "ack",
    ]:
        if method not in rows:
            continue
        scores = [rows[method].get(d) for d in all_datasets]
        valid_scores = [s for s in scores if s is not None]
        avg = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None
        method_memory = memory.get(method, {})
        peak_values = [
            method_memory[d]["peak_gb"] for d in all_datasets
            if d in method_memory and method_memory[d].get("peak_gb") is not None
        ]
        kv_values = [
            method_memory[d]["kv_cache_mb"] for d in all_datasets
            if d in method_memory and method_memory[d].get("kv_cache_mb") is not None
        ]
        avg_latency_values = [
            method_memory[d]["avg_latency_s"] for d in all_datasets
            if d in method_memory and method_memory[d].get("avg_latency_s") is not None
        ]
        avg_prefill_latency_values = [
            method_memory[d]["avg_prefill_latency_s"] for d in all_datasets
            if d in method_memory and method_memory[d].get("avg_prefill_latency_s") is not None
        ]
        avg_decode_latency_values = [
            method_memory[d]["avg_decode_latency_s"] for d in all_datasets
            if d in method_memory and method_memory[d].get("avg_decode_latency_s") is not None
        ]
        max_prefill_latency_values = [
            method_memory[d]["max_prefill_latency_s"] for d in all_datasets
            if d in method_memory and method_memory[d].get("max_prefill_latency_s") is not None
        ]
        max_decode_latency_values = [
            method_memory[d]["max_decode_latency_s"] for d in all_datasets
            if d in method_memory and method_memory[d].get("max_decode_latency_s") is not None
        ]
        throughput_values = [
            method_memory[d]["tokens_per_second"] for d in all_datasets
            if d in method_memory and method_memory[d].get("tokens_per_second") is not None
        ]
        profiled_tflops_values = [
            method_memory[d]["profiled_tflops"] for d in all_datasets
            if d in method_memory and method_memory[d].get("profiled_tflops") is not None
        ]
        profiled_tflops_per_s_values = [
            method_memory[d]["profiled_tflops_per_s"] for d in all_datasets
            if d in method_memory and method_memory[d].get("profiled_tflops_per_s") is not None
        ]
        peak_gb = round(max(peak_values), 2) if peak_values else ""
        kv_cache_mb = round(max(kv_values), 1) if kv_values else ""
        avg_latency_s = round(sum(avg_latency_values) / len(avg_latency_values), 3) if avg_latency_values else ""
        avg_prefill_latency_s = round(sum(avg_prefill_latency_values) / len(avg_prefill_latency_values), 3) if avg_prefill_latency_values else ""
        avg_decode_latency_s = round(sum(avg_decode_latency_values) / len(avg_decode_latency_values), 3) if avg_decode_latency_values else ""
        max_prefill_latency_s = round(sum(max_prefill_latency_values) / len(max_prefill_latency_values), 3) if max_prefill_latency_values else ""
        max_decode_latency_s = round(sum(max_decode_latency_values) / len(max_decode_latency_values), 3) if max_decode_latency_values else ""
        tokens_per_second = round(sum(throughput_values) / len(throughput_values), 2) if throughput_values else ""
        profiled_tflops = round(sum(profiled_tflops_values) / len(profiled_tflops_values), 6) if profiled_tflops_values else ""
        profiled_tflops_per_s = round(sum(profiled_tflops_per_s_values) / len(profiled_tflops_per_s_values), 6) if profiled_tflops_per_s_values else ""
        row = (
            [method]
            + [round(s, 2) if s is not None else "" for s in scores]
            + [avg if avg is not None else ""]
            + [peak_gb]
            + [kv_cache_mb]
            + [avg_latency_s]
            + [avg_prefill_latency_s]
            + [avg_decode_latency_s]
            + [max_prefill_latency_s]
            + [max_decode_latency_s]
            + [tokens_per_second]
            + [profiled_tflops]
            + [profiled_tflops_per_s]
        )
        writer.writerow(row)

    csv_content = output.getvalue()
    print("\n=== RESULTS CSV ===")
    print(csv_content)

    # Save to volume
    csv_path = f"{results_dir}/summary.csv"
    with open(csv_path, "w") as f:
        f.write(csv_content)
    print(f"Saved to {csv_path}")

    return csv_content


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=900
)
def generate_xgb_router_dataset(run_tag: str, methods: list[str] | None = None, latency_weight: float = 0.0):
    import json, os, sys
    from datasets import load_dataset

    sys.path.insert(0, "/app/experiments/LongBench")
    from eval import dataset2metric

    methods_to_use = methods or XGB_ROUTER_CANDIDATE_METHODS
    config_dir = "/app/experiments/LongBench/config"
    with open(os.path.join(config_dir, "dataset2prompt.json")) as f:
        dataset2prompt = json.load(f)
    with open(os.path.join(config_dir, "dataset2maxlen.json")) as f:
        dataset2maxlen = json.load(f)

    code_markers = (
        "def ", "class ", "import ", "from ", "return ", "self.", "->", "::",
        "();", "</", "#include", "public ", "private ",
    )
    summary_markers = (
        "summary:", "summarize", "write a summary", "write a one-page summary",
    )
    qa_markers = (
        "question:", "answer:", "passage", "given passages", "given a scientific article",
    )

    def prompt_features(prompt: str, max_gen: int):
        sample = (prompt[:512] + "\n" + prompt[-512:]).lower()
        code_hits = sum(marker in sample for marker in code_markers)
        summary_hits = sum(marker in sample for marker in summary_markers)
        qa_hits = sum(marker in sample for marker in qa_markers)
        return {
            "max_gen": max_gen,
            "prompt_chars": len(prompt),
            "sample_chars": len(sample),
            "code_marker_hits": code_hits,
            "summary_marker_hits": summary_hits,
            "qa_marker_hits": qa_hits,
            "newline_ratio": (prompt.count("\n") / max(len(prompt), 1)),
            "digit_ratio": (sum(ch.isdigit() for ch in prompt) / max(len(prompt), 1)),
            "punct_ratio": (sum((not ch.isalnum()) and (not ch.isspace()) for ch in prompt) / max(len(prompt), 1)),
        }

    def load_predictions(method: str, dataset: str):
        model_name = METHODS[method]["model_name"]
        path = f"{_predictions_dir(run_tag, method)}/pred_e/{model_name}/{dataset}.jsonl"
        if not os.path.exists(path):
            path = f"{_predictions_dir(run_tag, method)}/pred/{model_name}/{dataset}.jsonl"
        if not os.path.exists(path):
            return None
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def load_avg_latency(method: str, dataset: str):
        path = f"{_results_dir(run_tag)}/{method}_{dataset}_memory.json"
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f).get("avg_latency_s")

    def score_prediction(dataset: str, pred: str, answers: list[str], all_classes):
        score = 0.0
        metric = dataset2metric[dataset]
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip("\n").split("\n")[0]
        for answer in answers:
            score = max(score, metric(pred, answer, all_classes=all_classes))
        return round(100 * score, 6)

    out_dir = f"{_results_dir(run_tag)}/router_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/xgb_candidates.jsonl"
    summary_path = f"{out_dir}/xgb_candidates_summary.json"

    total_records = 0
    missing = []
    best_counts = {}
    with open(out_path, "w", encoding="utf-8") as out_f:
        for dataset in DATASETS:
            data = list(load_dataset("THUDM/LongBench", dataset, split="test", trust_remote_code=True))
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            method_rows = {method: load_predictions(method, dataset) for method in methods_to_use}
            for method, rows in method_rows.items():
                if rows is None:
                    missing.append({"method": method, "dataset": dataset, "reason": "missing_predictions"})

            complete_methods = [method for method, rows in method_rows.items() if rows is not None]
            if not complete_methods:
                continue

            avg_latencies = {method: load_avg_latency(method, dataset) for method in complete_methods}
            row_count = min(len(method_rows[method]) for method in complete_methods)
            for idx in range(min(len(data), row_count)):
                prompt = prompt_format.format(**data[idx])
                candidates = {}
                for method in complete_methods:
                    pred_row = method_rows[method][idx]
                    latency_s = pred_row.get("latency_s")
                    if latency_s is None:
                        latency_s = avg_latencies.get(method)
                    score = score_prediction(
                        dataset,
                        pred_row.get("pred", ""),
                        pred_row.get("answers", []),
                        pred_row.get("all_classes"),
                    )
                    utility = score - latency_weight * latency_s if latency_s is not None else score
                    candidates[method] = {
                        "score": score,
                        "utility": utility,
                        "latency_s": latency_s,
                        "prefill_latency_s": pred_row.get("prefill_latency_s"),
                        "decode_latency_s": pred_row.get("decode_latency_s"),
                        "generated_tokens": pred_row.get("generated_tokens"),
                        "context_length": pred_row.get("context_length"),
                        "pred": pred_row.get("pred", ""),
                    }

                best_score_method = max(candidates, key=lambda method: candidates[method]["score"])
                best_utility_method = max(candidates, key=lambda method: candidates[method]["utility"])
                best_counts[best_utility_method] = best_counts.get(best_utility_method, 0) + 1
                record = {
                    "run_tag": run_tag,
                    "dataset": dataset,
                    "example_idx": idx,
                    "length": data[idx].get("length"),
                    "features": prompt_features(prompt, max_gen),
                    "candidate_methods": complete_methods,
                    "candidates": candidates,
                    "best_score_method": best_score_method,
                    "best_utility_method": best_utility_method,
                    "latency_weight": latency_weight,
                }
                json.dump(record, out_f, ensure_ascii=False)
                out_f.write("\n")
                total_records += 1

    summary = {
        "run_tag": run_tag,
        "methods": methods_to_use,
        "datasets": DATASETS,
        "records": total_records,
        "latency_weight": latency_weight,
        "best_utility_counts": best_counts,
        "missing": missing,
        "output": out_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved XGBoost router dataset to {out_path}")
    print(f"Saved XGBoost router summary to {summary_path}")
    print(json.dumps(summary, indent=2))
    return summary


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=900
)
def train_xgb_router(run_tag: str, label_field: str = "best_utility_method", include_dataset: bool = False):
    import json, os
    from collections import Counter

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    router_dir = f"{_results_dir(run_tag)}/router_data"
    data_path = f"{router_dir}/xgb_candidates.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing router data: {data_path}")

    feature_names = [
        "max_gen",
        "prompt_chars",
        "sample_chars",
        "code_marker_hits",
        "summary_marker_hits",
        "qa_marker_hits",
        "newline_ratio",
        "digit_ratio",
        "punct_ratio",
        "length",
    ]
    dataset_names = DATASETS if include_dataset else []
    feature_names += [f"dataset__{dataset}" for dataset in dataset_names]

    X = []
    y_labels = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            label = record.get(label_field)
            if not label:
                continue
            features = record.get("features", {})
            row = [
                float(features.get("max_gen") or 0),
                float(features.get("prompt_chars") or 0),
                float(features.get("sample_chars") or 0),
                float(features.get("code_marker_hits") or 0),
                float(features.get("summary_marker_hits") or 0),
                float(features.get("qa_marker_hits") or 0),
                float(features.get("newline_ratio") or 0),
                float(features.get("digit_ratio") or 0),
                float(features.get("punct_ratio") or 0),
                float(record.get("length") or 0),
            ]
            if include_dataset:
                row.extend(1.0 if record.get("dataset") == dataset else 0.0 for dataset in DATASETS)
            X.append(row)
            y_labels.append(label)

    if not X:
        raise ValueError(f"No usable rows found in {data_path}")

    label_names = sorted(set(y_labels))
    if len(label_names) < 2:
        raise ValueError(f"Need at least two router labels to train XGBoost, found: {label_names}")
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y = np.array([label_to_id[label] for label in y_labels], dtype=np.int64)
    X = np.array(X, dtype=np.float32)

    stratify = y if min(Counter(y).values()) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_names),
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = float(accuracy_score(y_val, y_pred))
    report = classification_report(
        y_val,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    model_path = f"{router_dir}/xgb_router.json"
    metadata_path = f"{router_dir}/xgb_router_metadata.json"
    metrics_path = f"{router_dir}/xgb_router_metrics.json"
    model.save_model(model_path)

    metadata = {
        "run_tag": run_tag,
        "label_field": label_field,
        "include_dataset": include_dataset,
        "feature_names": feature_names,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "model_path": model_path,
        "rows": int(X.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
    }
    metrics = {
        "accuracy": accuracy,
        "label_counts": dict(Counter(y_labels)),
        "classification_report": report,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved XGBoost router model to {model_path}")
    print(f"Saved XGBoost router metadata to {metadata_path}")
    print(f"Saved XGBoost router metrics to {metrics_path}")
    print(json.dumps({"accuracy": accuracy, "label_counts": metrics["label_counts"]}, indent=2))
    return {"model_path": model_path, "metadata_path": metadata_path, "metrics_path": metrics_path, "accuracy": accuracy}


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=120
)
def verify_eval_complete(run_tag: str, methods: list[str] | None = None):
    import os, json

    methods_to_check = methods or STATIC_METHODS
    results_dir = _results_dir(run_tag)
    missing = []
    missing_metrics = []
    required_metric_fields = [
        "peak_gb",
        "kv_cache_mb",
        "total_latency_s",
        "avg_latency_s",
        "max_latency_s",
        "tokens_per_second",
        "profiled_flops",
        "profiled_tflops",
        "profiled_tflops_per_s",
    ]

    for method in methods_to_check:
        for dataset in DATASETS:
            path = os.path.join(results_dir, f"{method}_{dataset}.json")
            if not os.path.exists(path):
                missing.append((method, dataset))
            metric_path = os.path.join(results_dir, f"{method}_{dataset}_memory.json")
            if not os.path.exists(metric_path):
                missing_metrics.append((method, dataset, "missing memory json"))
                continue
            with open(metric_path) as f:
                metric_data = json.load(f)
            absent_fields = [
                field for field in required_metric_fields
                if metric_data.get(field) is None
            ]
            if absent_fields:
                missing_metrics.append((method, dataset, ", ".join(absent_fields)))

    if missing:
        print(f"Eval incomplete for run {run_tag}. Missing {len(missing)} result files.")
        for method, dataset in missing[:20]:
            print(f"- missing: {method} / {dataset}")
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more")
        return False

    if missing_metrics:
        print(f"Eval incomplete for run {run_tag}. Missing logged metrics for {len(missing_metrics)} method/dataset pairs.")
        for method, dataset, detail in missing_metrics[:20]:
            print(f"- metrics missing: {method} / {dataset} -> {detail}")
        if len(missing_metrics) > 20:
            print(f"... and {len(missing_metrics) - 20} more")
        return False

    print(f"EVAL PASSED for run {run_tag}")
    return True


# ============================================================
# Entrypoints
# ============================================================

def _submit_inference_methods(methods: list[str], version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    submit_inference_batch.spawn(methods, resolved_run_tag)
    print(f"Submitted remote inference orchestrator for {len(methods)} method(s).")


def _submit_limited_inference_methods(methods: list[str], version: str = "1", run_tag: str = "", limit: int = XGB_ROUTER_SAMPLE_LIMIT, sample_offset: int = 0):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    submit_inference_limited_batch.spawn(methods, resolved_run_tag, limit, sample_offset)
    print(f"Submitted remote limited inference orchestrator for {len(methods)} method(s), limit={limit}, offset={sample_offset}.")


def _submit_eval_methods(methods: list[str], run_tag: str):
    submit_eval_batch.spawn(methods, run_tag)
    print(f"Submitted remote eval orchestrator for {len(methods)} method(s).")


def _submit_validation_methods(methods: list[str], version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    submit_validation_batch.spawn(
        methods,
        resolved_run_tag,
        VALIDATION_DATASET,
        VALIDATION_SAMPLE_OFFSET,
    )
    print(f"Submitted remote validation orchestrator for {len(methods)} method(s).")

@app.local_entrypoint()
def main(version: str = "1", run_tag: str = ""):
    """Run inference for all methods and datasets via a remote orchestrator."""
    _submit_inference_methods(STATIC_METHODS, version, run_tag)


@app.local_entrypoint()
def main_eval_all(run_tag: str):
    """Score all completed inference runs via a remote orchestrator."""
    _submit_eval_methods(STATIC_METHODS, run_tag)


@app.local_entrypoint()
def main_csv(run_tag: str):
    """Generate results CSV from saved eval results."""
    generate_csv.spawn(run_tag)


@app.local_entrypoint()
def main_verify_eval(run_tag: str):
    """Check whether all static-method eval artifacts exist for the given run."""
    verify_eval_complete.spawn(run_tag)

@app.local_entrypoint()
def main_validate_all_static(version: str = "1", run_tag: str = ""):
    """Run one-example validation for all current static methods."""
    _submit_validation_methods(STATIC_METHODS, version, run_tag)


@app.local_entrypoint()
def main_dynamic(version: str = "1", run_tag: str = ""):
    """Run inference for all current dynamic methods and datasets via a remote orchestrator."""
    _submit_inference_methods(DYNAMIC_METHODS, version, run_tag)


@app.local_entrypoint()
def main_eval_all_dynamic(run_tag: str):
    """Score all completed dynamic-method inference runs via a remote orchestrator."""
    _submit_eval_methods(DYNAMIC_METHODS, run_tag)


@app.local_entrypoint()
def main_verify_eval_dynamic(run_tag: str):
    """Check whether all dynamic-method eval artifacts exist for the given run."""
    verify_eval_complete.spawn(run_tag, DYNAMIC_METHODS)


@app.local_entrypoint()
def main_validate_all_dynamic(version: str = "1", run_tag: str = ""):
    """Run one-example validation for all current dynamic methods."""
    _submit_validation_methods(DYNAMIC_METHODS, version, run_tag)


@app.local_entrypoint()
def main_xgb_router_data(version: str = "1", run_tag: str = ""):
    """Run inference for the XGBoost router candidate methods."""
    _submit_inference_methods(XGB_ROUTER_CANDIDATE_METHODS, version, run_tag)


@app.local_entrypoint()
def main_xgb_router_data_100(version: str = "1", run_tag: str = "", sample_offset: int = 0):
    """Run 100 examples per dataset for the XGBoost router candidate methods."""
    _submit_limited_inference_methods(
        XGB_ROUTER_CANDIDATE_METHODS,
        version,
        run_tag,
        XGB_ROUTER_SAMPLE_LIMIT,
        sample_offset,
    )


@app.local_entrypoint()
def main_eval_xgb_router_data(run_tag: str):
    """Score the XGBoost router candidate-method predictions."""
    _submit_eval_methods(XGB_ROUTER_CANDIDATE_METHODS, run_tag)


@app.local_entrypoint()
def main_verify_eval_xgb_router_data(run_tag: str):
    """Check eval artifacts for the XGBoost router candidate methods."""
    verify_eval_complete.spawn(run_tag, XGB_ROUTER_CANDIDATE_METHODS)


@app.local_entrypoint()
def main_build_xgb_router_data(run_tag: str, latency_weight: float = 0.0):
    """Build per-example XGBoost router training rows from candidate predictions."""
    generate_xgb_router_dataset.spawn(run_tag, XGB_ROUTER_CANDIDATE_METHODS, latency_weight)


@app.local_entrypoint()
def main_train_xgb_router(run_tag: str, label_field: str = "best_utility_method", include_dataset: bool = False):
    """Train an XGBoost method router from built router_data/xgb_candidates.jsonl."""
    train_xgb_router.spawn(run_tag, label_field, include_dataset)


@app.local_entrypoint()
def main_xgb_routing(router_run_tag: str, version: str = "1", run_tag: str = ""):
    """Run the trained XGBoost router across all datasets."""
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    print(f"Router run tag: {router_run_tag}")
    submit_xgb_routing_batch.spawn(resolved_run_tag, router_run_tag)


@app.local_entrypoint()
def main_eval_xgb_routing(run_tag: str):
    """Eval trained XGBoost-router predictions."""
    _submit_eval_methods(["xgb_routing"], run_tag)


@app.local_entrypoint()
def main_verify_eval_xgb_routing(run_tag: str):
    """Check trained XGBoost-router eval artifacts."""
    verify_eval_complete.spawn(run_tag, ["xgb_routing"])


@app.local_entrypoint()
def main_validate_xgb_routing(router_run_tag: str, version: str = "1", run_tag: str = ""):
    """Run one-example validation for the trained XGBoost router."""
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    print(f"Router run tag: {router_run_tag}")
    run_validation.spawn("xgb_routing", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag, router_run_tag)


@app.local_entrypoint()
def main_validate_xgb_router_data(version: str = "1", run_tag: str = ""):
    """Run one-example validation for the XGBoost router candidate methods."""
    _submit_validation_methods(XGB_ROUTER_CANDIDATE_METHODS, version, run_tag)


@app.local_entrypoint()
def main_validate_single(version: str = "1", run_tag: str = ""):
    """Run one-example validation for a single method. Edit method below."""
    method = "clusterkv_quest_bounds_static"
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn(method, VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)
    print("Submitted 1 validation job (spawn).")


@app.local_entrypoint()
def main_single(version: str = "1", run_tag: str = ""):
    """Run a single method/dataset combination. Edit method/dataset below."""
    method = "baseline"
    dataset = "hotpotqa"
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_inference.spawn(method, dataset, resolved_run_tag)


@app.local_entrypoint()
def main_eval_single(run_tag: str):
    """Eval a single method/dataset combination. Edit method/dataset below."""
    method = "baseline"
    dataset = "hotpotqa"
    run_eval.spawn(method, dataset, run_tag)


@app.local_entrypoint()
def main_method(method: str, version: str = "1", run_tag: str = ""):
    """Run any single registered method across all datasets."""
    if not _is_registered_method(method):
        raise ValueError(f"Unknown method: {method}")
    _submit_inference_methods([method], version, run_tag)


@app.local_entrypoint()
def main_eval_method(method: str, run_tag: str):
    """Eval any single registered method across all datasets."""
    if not _is_registered_method(method):
        raise ValueError(f"Unknown method: {method}")
    _submit_eval_methods([method], run_tag)


@app.local_entrypoint()
def main_verify_eval_method(method: str, run_tag: str):
    """Check eval artifacts for any single registered method."""
    if not _is_registered_method(method):
        raise ValueError(f"Unknown method: {method}")
    verify_eval_complete.spawn(run_tag, [method])


@app.local_entrypoint()
def main_validate_method(method: str, version: str = "1", run_tag: str = ""):
    """Run one-example validation for any single registered method."""
    if not _is_registered_method(method):
        raise ValueError(f"Unknown method: {method}")
    _submit_validation_methods([method], version, run_tag)


@app.local_entrypoint()
def main_heuristic_routing(version: str = "1", run_tag: str = ""):
    """Run routed heuristic inference across all datasets."""
    _submit_inference_methods(["heuristic_routing"], version, run_tag)


@app.local_entrypoint()
def main_eval_heuristic_routing(run_tag: str):
    """Eval routed heuristic predictions across all datasets."""
    _submit_eval_methods(["heuristic_routing"], run_tag)


@app.local_entrypoint()
def main_verify_eval_heuristic_routing(run_tag: str):
    """Check routed heuristic eval artifacts."""
    verify_eval_complete.spawn(run_tag, ["heuristic_routing"])


@app.local_entrypoint()
def main_validate_heuristic_routing(version: str = "1", run_tag: str = ""):
    """Run one-example validation for the routed heuristic."""
    _submit_validation_methods(["heuristic_routing"], version, run_tag)


# Convenience entrypoints for methods you've already run
@app.local_entrypoint()
def main_baseline(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("baseline", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_baseline_clusterpath_static(version: str = "1", run_tag: str = ""):
    """Run baseline through the generalized ClusterKV path with compression disabled."""
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("baseline_clusterpath_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_baseline_clusterpath_static_a100_80gb(version: str = "1", run_tag: str = ""):
    """Run the no-compression cluster-path baseline control on A100-80GB."""
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference_a100_80gb.spawn("baseline_clusterpath_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_snapkv(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_quest(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("quest_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_quest_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("quest_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterattn_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterattn_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterattn_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterattn_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterattn_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterattn_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterattn_expected_attention_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterattn_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_quest_bounds(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_quest_bounds_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_quest_bounds_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_quest_bounds_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_quest_bounds_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_snapkv(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_snapkv_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_snapkv_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_snapkv_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_snapkv_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_h2o(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_h2o_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_h2o_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_h2o_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_h2o_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_recon(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_recon_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_recon_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_recon_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_recon_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_expected_attention_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_expected_attention_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_expected_attention_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_expected_attention_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_expected_attention_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_random(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_random_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_random_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_random_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("clusterkv_random_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("pagekv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("pagekv_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("pagekv_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("pagekv_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("pagekv_expected_attention_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("pagekv_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("tokenkv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("tokenkv_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("tokenkv_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("tokenkv_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("tokenkv_expected_attention_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("tokenkv_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_validate_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("snapkv_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_quest_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("quest_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterattn_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterattn_quest_bounds_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterattn_snapkv_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterattn_h2o_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterattn_recon_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterattn_expected_attention_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterattn_random_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_pagekv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("pagekv_expected_attention_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_tokenkv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("tokenkv_expected_attention_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_quest_bounds_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_quest_bounds_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_quest_bounds_kmeans_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_quest_bounds_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_quest_bounds_spherical_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_snapkv_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_snapkv_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_snapkv_kmeans_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_snapkv_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_snapkv_spherical_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_h2o_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_h2o_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_h2o_kmeans_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_h2o_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_h2o_spherical_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_recon_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_recon_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_recon_kmeans_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_recon_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_recon_spherical_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_expected_attention_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_expected_attention_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_expected_attention_kmeans_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_expected_attention_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_expected_attention_spherical_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_random_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_random_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_random_kmeans_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterkv_random_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.spawn("clusterkv_random_spherical_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_eval_baseline(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("baseline", dataset, run_tag)

@app.local_entrypoint()
def main_eval_baseline_clusterpath_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("baseline_clusterpath_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_snapkv(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_quest(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("quest_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_quest_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("quest_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterattn_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterattn_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_quest_bounds_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterattn_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterattn_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_h2o_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterattn_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_recon_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterattn_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_expected_attention_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterattn_expected_attention_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_random_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterattn_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_quest_bounds_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_quest_bounds_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_snapkv_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_snapkv_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_h2o(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_h2o_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_h2o_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_h2o_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_h2o_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_h2o_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_recon(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_recon_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_recon_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_recon_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_recon_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_recon_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_expected_attention_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_expected_attention_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_expected_attention_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_expected_attention_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_expected_attention_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_expected_attention_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_random(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_random_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_random_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_random_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_random_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("clusterkv_random_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_quest_bounds_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("pagekv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("pagekv_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_h2o_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("pagekv_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_recon_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("pagekv_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_expected_attention_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("pagekv_expected_attention_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_random_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("pagekv_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_quest_bounds_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("tokenkv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("tokenkv_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_h2o_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("tokenkv_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_recon_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("tokenkv_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_expected_attention_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("tokenkv_expected_attention_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_random_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.spawn("tokenkv_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_topk_ablation(
    version: str = "1",
    run_tag: str = "",
    base_methods_csv: str = "",
    k_rets_csv: str = "",
    limit: int | None = None,
    sample_offset: int = 0,
):
    """
    Sweep max_capacity_prompt via k_ret = max_capacity_prompt - window_size for each base method.

    Each job uses synthetic method id "{base}__kret{k_ret}" (eval/validate with the same id).
    Override defaults: --base-methods-csv clusterattn_h2o_static,tokenkv_snapkv_static
    --k-rets-csv 256,512,1024,2048
    """
    resolved_run_tag = _build_run_tag(version, run_tag)
    bases = [b.strip() for b in base_methods_csv.split(",") if b.strip()]
    if not bases:
        bases = list(TOPK_ABLATION_DEFAULT_BASE_METHODS)
    k_list = [int(x.strip()) for x in k_rets_csv.split(",") if x.strip()]
    if not k_list:
        k_list = list(TOPK_ABLATION_DEFAULT_K_RETS)
    for b in bases:
        if b not in METHODS:
            raise ValueError(f"Unknown base method {b!r}; must be a key in METHODS.")
        extra = METHODS[b]["extra_args"]
        if "--compress_args_path" not in extra:
            raise ValueError(
                f"{b!r} has no --compress_args_path in extra_args; "
                "top-k ablation only applies to methods that load a LongBench JSON config."
            )
    print(f"Run tag: {resolved_run_tag}")
    print(f"Top-k ablation: bases={bases} k_ret list={k_list} datasets={DATASETS}")
    submitted = 0
    for base_method in bases:
        for k_ret in k_list:
            for dataset in DATASETS:
                run_inference_topk.spawn(
                    base_method, k_ret, dataset, resolved_run_tag, limit, sample_offset
                )
                submitted += 1
    print(f"Submitted {submitted} top-k ablation inference jobs.")


@app.local_entrypoint()
def main_h2o(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("h2o_static", dataset, resolved_run_tag)
