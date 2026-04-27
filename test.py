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

    peak_gb = float(peak_match.group(2)) if peak_match else None
    kv_cache_mb = float(kv_match.group(1)) if kv_match else None

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
            }

    if not rows:
        print(f"No scored results found yet for run {run_tag}.")
        return

    # Build CSV
    all_datasets = sorted({d for m in rows.values() for d in m})
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Method"] + all_datasets + ["Average", "Peak GPU (GB)", "KV Cache (MB)"])

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
        "tokenkv_quest_bounds_static",
        "tokenkv_snapkv_static",
        "tokenkv_h2o_static",
        "tokenkv_recon_static",
        "tokenkv_expected_attention_static",
        "tokenkv_random_static",
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
        peak_gb = round(max(peak_values), 2) if peak_values else ""
        kv_cache_mb = round(max(kv_values), 1) if kv_values else ""
        row = (
            [method]
            + [round(s, 2) if s is not None else "" for s in scores]
            + [avg if avg is not None else ""]
            + [peak_gb]
            + [kv_cache_mb]
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
    import os

    methods_to_check = methods or STATIC_METHODS
    results_dir = _results_dir(run_tag)
    missing = []

    for method in methods_to_check:
        for dataset in DATASETS:
            path = os.path.join(results_dir, f"{method}_{dataset}.json")
            if not os.path.exists(path):
                missing.append((method, dataset))

    if missing:
        print(f"Eval incomplete for run {run_tag}. Missing {len(missing)} result files.")
        for method, dataset in missing[:20]:
            print(f"- missing: {method} / {dataset}")
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more")
        return False

    print(f"EVAL PASSED for run {run_tag}")
    return True


# ============================================================
# Entrypoints
# ============================================================

@app.local_entrypoint()
def main(version: str = "1", run_tag: str = ""):
    """Run inference for all methods and datasets, then eval and generate CSV."""
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    methods_to_run = STATIC_METHODS
    for method in methods_to_run:
        for dataset in DATASETS:
            print(f"\nSubmitting {method} / {dataset}...")
            run_inference.remote(method, dataset, resolved_run_tag)
    print(f"Submitted inference jobs for {len(methods_to_run)} methods across {len(DATASETS)} datasets.")


@app.local_entrypoint()
def main_eval_all(run_tag: str):
    """Score all completed inference runs and generate CSV."""
    methods_to_run = STATIC_METHODS
    for method in methods_to_run:
        for dataset in DATASETS:
            print(f"Evaluating {method} / {dataset}...")
            run_eval.remote(method, dataset, run_tag)
    print(f"Submitted eval jobs for {len(methods_to_run)} methods across {len(DATASETS)} datasets.")


@app.local_entrypoint()
def main_csv(run_tag: str):
    """Generate results CSV from saved eval results."""
    generate_csv.remote(run_tag)


@app.local_entrypoint()
def main_verify_eval(run_tag: str):
    """Check whether all static-method eval artifacts exist for the given run."""
    verify_eval_complete.remote(run_tag)

@app.local_entrypoint()
def main_validate_all_static(version: str = "1", run_tag: str = ""):
    """Run one-example validation for all current static methods."""
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for method in STATIC_METHODS:
        run_validation.spawn(method, VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)
    print(f"Submitted {len(STATIC_METHODS)} validation jobs (spawn).")

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
    run_inference.remote(method, dataset, resolved_run_tag)


@app.local_entrypoint()
def main_eval_single(run_tag: str):
    """Eval a single method/dataset combination. Edit method/dataset below."""
    method = "baseline"
    dataset = "hotpotqa"
    run_eval.remote(method, dataset, run_tag)


# Convenience entrypoints for methods you've already run
@app.local_entrypoint()
def main_baseline(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("baseline", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_snapkv(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_quest(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("quest_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_quest_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("quest_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterattn_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterattn_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterattn_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterattn_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterattn_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterattn_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterattn_expected_attention_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterattn_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterattn_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_quest_bounds(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_quest_bounds_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_quest_bounds_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_quest_bounds_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_quest_bounds_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_snapkv(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_snapkv_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_snapkv_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_snapkv_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_snapkv_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_h2o(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_h2o_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_h2o_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_h2o_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_h2o_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_recon(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_recon_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_recon_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_recon_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_recon_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_expected_attention_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_expected_attention_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_expected_attention_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_expected_attention_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_expected_attention_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_random(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_random_kmeans_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_random_kmeans_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_clusterkv_random_spherical_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("clusterkv_random_spherical_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("pagekv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("pagekv_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("pagekv_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("pagekv_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("pagekv_expected_attention_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_pagekv_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("pagekv_random_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("tokenkv_quest_bounds_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("tokenkv_snapkv_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("tokenkv_h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("tokenkv_recon_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("tokenkv_expected_attention_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_tokenkv_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("tokenkv_random_static", dataset, resolved_run_tag)

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
    run_validation.remote("clusterattn_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_quest_bounds_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.remote("clusterattn_quest_bounds_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_snapkv_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.remote("clusterattn_snapkv_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.remote("clusterattn_h2o_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_recon_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.remote("clusterattn_recon_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_expected_attention_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.remote("clusterattn_expected_attention_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

@app.local_entrypoint()
def main_validate_clusterattn_random_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    run_validation.remote("clusterattn_random_static", VALIDATION_DATASET, VALIDATION_SAMPLE_OFFSET, resolved_run_tag)

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
        run_eval.remote("baseline", dataset, run_tag)

@app.local_entrypoint()
def main_eval_snapkv(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_quest(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("quest_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_quest_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("quest_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterattn_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterattn_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_quest_bounds_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterattn_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterattn_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_h2o_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterattn_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_recon_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterattn_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_expected_attention_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterattn_expected_attention_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterattn_random_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterattn_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_quest_bounds_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_quest_bounds_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_snapkv_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_snapkv_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_h2o(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_h2o_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_h2o_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_h2o_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_h2o_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_h2o_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_recon(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_recon_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_recon_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_recon_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_recon_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_recon_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_expected_attention_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_expected_attention_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_expected_attention_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_expected_attention_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_expected_attention_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_expected_attention_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_random(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_random_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_random_kmeans_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_random_kmeans_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_clusterkv_random_spherical_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("clusterkv_random_spherical_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_quest_bounds_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("pagekv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("pagekv_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_h2o_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("pagekv_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_recon_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("pagekv_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_expected_attention_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("pagekv_expected_attention_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_pagekv_random_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("pagekv_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_quest_bounds_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("tokenkv_quest_bounds_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_snapkv_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("tokenkv_snapkv_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_h2o_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("tokenkv_h2o_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_recon_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("tokenkv_recon_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_expected_attention_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("tokenkv_expected_attention_static", dataset, run_tag)

@app.local_entrypoint()
def main_eval_tokenkv_random_static(run_tag: str):
    for dataset in DATASETS:
        run_eval.remote("tokenkv_random_static", dataset, run_tag)

@app.local_entrypoint()
def main_h2o(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("h2o_static", dataset, resolved_run_tag)

@app.local_entrypoint()
def main_h2o_static(version: str = "1", run_tag: str = ""):
    resolved_run_tag = _build_run_tag(version, run_tag)
    print(f"Run tag: {resolved_run_tag}")
    for dataset in DATASETS:
        run_inference.remote("h2o_static", dataset, resolved_run_tag)
