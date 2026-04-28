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
    .pip_install("transformers==4.37.0")
    .pip_install(
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu121torch2.4-cp311-cp311-linux_x86_64.whl"
    )
    .pip_install("jieba", "rouge", "fuzzywuzzy", "python-Levenshtein")
    .add_local_dir(".", remote_path="/app", copy=True, ignore=[".venv"])
    .run_commands("cd /app && pip install -e .")
)

# ============================================================
# Datasets and methods config
# ============================================================

DATASETS = ["qasper", "hotpotqa", "gov_report", "lcc"]
VALIDATION_DATASET = "hotpotqa"
VALIDATION_SAMPLE_OFFSET = 0

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
    "clusterattn_dynamic": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "clusterattn_snapkv_dynamic_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2clusterattn_dynamic_c4096_w64_p16",
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
}

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
def run_inference(method: str, dataset: str, run_tag: str):
    import subprocess, shutil, os, re, json

    cfg = METHODS[method]
    cmd = [
        "python", cfg["script"],
        "--model", "mistral-7B-instruct-v0.2",
        "--dataset", dataset,
        "--write_model_name", cfg["model_name"],
    ] + cfg["extra_args"]

    print(f"\n{'='*50}")
    print(f"Running {method} on {dataset}")
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

    # Save predictions to volume
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
    avg_latency_match = re.search(r"Average latency / example:\s+([0-9.]+)\s+s", result.stdout)
    max_latency_match = re.search(r"Max latency / example:\s+([0-9.]+)\s+s", result.stdout)
    throughput_match = re.search(r"Generation throughput:\s+([0-9.]+)\s+tok/s", result.stdout)
    profiled_flops_match = re.search(r"Profiled FLOPs \(1st example\):\s+([0-9.]+)", result.stdout)
    profiled_tflops_match = re.search(r"Profiled TFLOPs \(1st example\):\s+([0-9.]+)", result.stdout)
    profiled_tflops_per_s_match = re.search(r"Profiled TFLOPs/s \(1st example\):\s+([0-9.]+)", result.stdout)

    peak_gb = float(peak_match.group(2)) if peak_match else None
    kv_cache_mb = float(kv_match.group(1)) if kv_match else None
    total_latency_s = float(total_latency_match.group(1)) if total_latency_match else None
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
                "returncode": result.returncode,
            },
            f,
        )
    print(f"Saved inference metadata to {metadata_path}")


@app.function(
    gpu="A100",
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=2400
)
def run_validation(method: str, dataset: str = VALIDATION_DATASET, sample_offset: int = VALIDATION_SAMPLE_OFFSET, run_tag: str = "unversioned"):
    import json, os, re, shutil, subprocess

    cfg = METHODS[method]
    cmd = [
        "python", cfg["script"],
        "--model", "mistral-7B-instruct-v0.2",
        "--dataset", dataset,
        "--limit", "1",
        "--sample_offset", str(sample_offset),
        "--write_model_name", cfg["model_name"],
    ] + cfg["extra_args"]

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

    cfg = METHODS[method]
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
            "Throughput (tok/s)",
            "Profiled TFLOPs",
            "Profiled TFLOPs/s",
        ]
    )

    for method in [
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
        "clusterattn_dynamic",
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
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    _submit_inference_methods([method], version, run_tag)


@app.local_entrypoint()
def main_eval_method(method: str, run_tag: str):
    """Eval any single registered method across all datasets."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    _submit_eval_methods([method], run_tag)


@app.local_entrypoint()
def main_validate_method(method: str, version: str = "1", run_tag: str = ""):
    """Run one-example validation for any single registered method."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    _submit_validation_methods([method], version, run_tag)


# Convenience entrypoints for methods you've already run
@app.local_entrypoint()
def main_baseline(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.spawn("baseline", dataset, resolved_run_tag)

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
