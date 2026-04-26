import modal

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
    .add_local_dir(".", remote_path="/app", copy=True)
    .run_commands("cd /app && pip install -e .")
)

# ============================================================
# Datasets and methods config
# ============================================================

DATASETS = ["qasper", "hotpotqa", "gov_report", "lcc"]

METHODS = {
    "baseline": {
        "script": "pred_snap.py",
        "extra_args": [],
        "model_name": "mistral-7B-instruct-v0.2",
    },
    "snapkv_static": {
        "script": "pred_snap.py",
        "extra_args": ["--compress_args_path", "ablation_c4096_w32_k7_maxpool.json"],
        "model_name": "mistral-7B-instruct-v0.2snapkv_static_c4096_w32_k7_maxpool",
    },
    "quest_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "quest",
            "--compress_args_path", "quest_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2quest_static_c4096_w64_p16",
    },
    "pagekv_quest_bounds_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_quest_bounds_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_quest_bounds_static_c4096_w64_p16",
    },
    "pagekv_snapkv_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_snapkv_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_snapkv_static_c4096_w64_p16",
    },
    "pagekv_h2o_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_h2o_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_h2o_static_c4096_w64_p16",
    },
    "pagekv_recon_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_recon_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_recon_static_c4096_w64_p16",
    },
    "pagekv_expected_attention_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_expected_attention_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_expected_attention_static_c4096_w64_p16",
    },
    "pagekv_random_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "pagekv_random_c4096_w64_p16.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2pagekv_random_static_c4096_w64_p16",
    },
    "tokenkv_quest_bounds_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_quest_bounds_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_quest_bounds_static_c4096_w64",
    },
    "tokenkv_snapkv_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_snapkv_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_snapkv_static_c4096_w64",
    },
    "tokenkv_h2o_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_h2o_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_h2o_static_c4096_w64",
    },
    "tokenkv_recon_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_recon_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_recon_static_c4096_w64",
    },
    "tokenkv_expected_attention_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_expected_attention_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_expected_attention_static_c4096_w64",
    },
    "tokenkv_random_static": {
        "script": "pred_snap.py",
        "extra_args": [
            "--method", "clusterkv",
            "--compress_args_path", "tokenkv_random_c4096_w64.json",
        ],
        "model_name": "mistral-7B-instruct-v0.2tokenkv_random_static_c4096_w64",
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
def run_inference(method: str, dataset: str):
    import subprocess, shutil, os, re, json

    cfg = METHODS[method]
    cmd = [
        "python", cfg["script"],
        "--model", "mistral-7B-instruct-v0.2",
        "--dataset", dataset,
    ] + cfg["extra_args"]

    print(f"\n{'='*50}")
    print(f"Running {method} on {dataset}")
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
    pred_dir = f"/models/predictions_{method}"
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

    os.makedirs("/models/results", exist_ok=True)
    memory_result_path = f"/models/results/{method}_{dataset}_memory.json"
    with open(memory_result_path, "w") as f:
        json.dump(
            {
                "method": method,
                "dataset": dataset,
                "peak_gb": peak_gb,
                "kv_cache_mb": kv_cache_mb,
            },
            f,
        )
    print(f"Saved memory stats to {memory_result_path}")


# ============================================================
# Eval
# ============================================================

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=300
)
def run_eval(method: str, dataset: str):
    import subprocess, shutil, os, json

    cfg = METHODS[method]
    model_name = cfg["model_name"]
    pred_dir = f"/models/predictions_{method}"

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
        os.makedirs("/models/results", exist_ok=True)
        volume_result_path = f"/models/results/{method}_{dataset}.json"
        with open(volume_result_path, "w") as f:
            json.dump({"method": method, "dataset": dataset, "score": score}, f)

    return score


# ============================================================
# CSV summary
# ============================================================

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=120
)
def generate_csv():
    import os, json, csv, io

    results_dir = "/models/results"
    if not os.path.exists(results_dir):
        print("No results found yet.")
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
        print("No results found yet.")
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
    csv_path = "/models/results/summary.csv"
    with open(csv_path, "w") as f:
        f.write(csv_content)
    print(f"Saved to {csv_path}")

    return csv_content


# ============================================================
# Entrypoints
# ============================================================

@app.local_entrypoint()
def main():
    """Run inference for all methods and datasets, then eval and generate CSV."""
    methods_to_run = ["baseline", "snapkv_static", "quest_static"]
    for method in methods_to_run:
        for dataset in DATASETS:
            print(f"\nSubmitting {method} / {dataset}...")
            run_inference.remote(method, dataset)


@app.local_entrypoint()
def main_eval_all():
    """Score all completed inference runs and generate CSV."""
    methods_to_run = ["baseline", "snapkv_static", "quest_static"]
    for method in methods_to_run:
        for dataset in DATASETS:
            print(f"Evaluating {method} / {dataset}...")
            run_eval.remote(method, dataset)


@app.local_entrypoint()
def main_csv():
    """Generate results CSV from saved eval results."""
    generate_csv.remote()


@app.local_entrypoint()
def main_single():
    """Run a single method/dataset combination. Edit method/dataset below."""
    method = "baseline"
    dataset = "hotpotqa"
    run_inference.remote(method, dataset)


@app.local_entrypoint()
def main_eval_single():
    """Eval a single method/dataset combination. Edit method/dataset below."""
    method = "baseline"
    dataset = "hotpotqa"
    run_eval.remote(method, dataset)


# Convenience entrypoints for methods you've already run
@app.local_entrypoint()
def main_baseline():
    for dataset in DATASETS:
        run_inference.remote("baseline", dataset)

@app.local_entrypoint()
def main_snapkv():
    for dataset in DATASETS:
        run_inference.remote("snapkv_static", dataset)

@app.local_entrypoint()
def main_snapkv_static():
    for dataset in DATASETS:
        run_inference.remote("snapkv_static", dataset)

@app.local_entrypoint()
def main_quest():
    for dataset in DATASETS:
        run_inference.remote("quest_static", dataset)

@app.local_entrypoint()
def main_quest_static():
    for dataset in DATASETS:
        run_inference.remote("quest_static", dataset)

@app.local_entrypoint()
def main_clusterkv():
    for dataset in DATASETS:
        run_inference.remote("pagekv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_clusterkv_quest_bounds():
    for dataset in DATASETS:
        run_inference.remote("pagekv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_clusterkv_quest_bounds_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_clusterkv_snapkv():
    for dataset in DATASETS:
        run_inference.remote("pagekv_snapkv_static", dataset)

@app.local_entrypoint()
def main_clusterkv_snapkv_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_snapkv_static", dataset)

@app.local_entrypoint()
def main_clusterkv_h2o():
    for dataset in DATASETS:
        run_inference.remote("pagekv_h2o_static", dataset)

@app.local_entrypoint()
def main_clusterkv_h2o_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_h2o_static", dataset)

@app.local_entrypoint()
def main_clusterkv_recon():
    for dataset in DATASETS:
        run_inference.remote("pagekv_recon_static", dataset)

@app.local_entrypoint()
def main_clusterkv_recon_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_recon_static", dataset)

@app.local_entrypoint()
def main_clusterkv_expected_attention_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_expected_attention_static", dataset)

@app.local_entrypoint()
def main_clusterkv_random():
    for dataset in DATASETS:
        run_inference.remote("pagekv_random_static", dataset)

@app.local_entrypoint()
def main_clusterkv_random_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_random_static", dataset)

@app.local_entrypoint()
def main_pagekv_quest_bounds_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_pagekv_snapkv_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_snapkv_static", dataset)

@app.local_entrypoint()
def main_pagekv_h2o_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_h2o_static", dataset)

@app.local_entrypoint()
def main_pagekv_recon_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_recon_static", dataset)

@app.local_entrypoint()
def main_pagekv_expected_attention_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_expected_attention_static", dataset)

@app.local_entrypoint()
def main_pagekv_random_static():
    for dataset in DATASETS:
        run_inference.remote("pagekv_random_static", dataset)

@app.local_entrypoint()
def main_tokenkv_quest_bounds_static():
    for dataset in DATASETS:
        run_inference.remote("tokenkv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_tokenkv_snapkv_static():
    for dataset in DATASETS:
        run_inference.remote("tokenkv_snapkv_static", dataset)

@app.local_entrypoint()
def main_tokenkv_h2o_static():
    for dataset in DATASETS:
        run_inference.remote("tokenkv_h2o_static", dataset)

@app.local_entrypoint()
def main_tokenkv_recon_static():
    for dataset in DATASETS:
        run_inference.remote("tokenkv_recon_static", dataset)

@app.local_entrypoint()
def main_tokenkv_expected_attention_static():
    for dataset in DATASETS:
        run_inference.remote("tokenkv_expected_attention_static", dataset)

@app.local_entrypoint()
def main_tokenkv_random_static():
    for dataset in DATASETS:
        run_inference.remote("tokenkv_random_static", dataset)

@app.local_entrypoint()
def main_eval_baseline():
    for dataset in DATASETS:
        run_eval.remote("baseline", dataset)

@app.local_entrypoint()
def main_eval_snapkv():
    for dataset in DATASETS:
        run_eval.remote("snapkv_static", dataset)

@app.local_entrypoint()
def main_eval_snapkv_static():
    for dataset in DATASETS:
        run_eval.remote("snapkv_static", dataset)

@app.local_entrypoint()
def main_eval_quest():
    for dataset in DATASETS:
        run_eval.remote("quest_static", dataset)

@app.local_entrypoint()
def main_eval_quest_static():
    for dataset in DATASETS:
        run_eval.remote("quest_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv():
    for dataset in DATASETS:
        run_eval.remote("pagekv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds():
    for dataset in DATASETS:
        run_eval.remote("pagekv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_quest_bounds_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv():
    for dataset in DATASETS:
        run_eval.remote("pagekv_snapkv_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_snapkv_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_snapkv_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_h2o():
    for dataset in DATASETS:
        run_eval.remote("pagekv_h2o_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_h2o_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_h2o_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_recon():
    for dataset in DATASETS:
        run_eval.remote("pagekv_recon_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_recon_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_recon_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_expected_attention_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_expected_attention_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_random():
    for dataset in DATASETS:
        run_eval.remote("pagekv_random_static", dataset)

@app.local_entrypoint()
def main_eval_clusterkv_random_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_random_static", dataset)

@app.local_entrypoint()
def main_eval_pagekv_quest_bounds_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_eval_pagekv_snapkv_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_snapkv_static", dataset)

@app.local_entrypoint()
def main_eval_pagekv_h2o_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_h2o_static", dataset)

@app.local_entrypoint()
def main_eval_pagekv_recon_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_recon_static", dataset)

@app.local_entrypoint()
def main_eval_pagekv_expected_attention_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_expected_attention_static", dataset)

@app.local_entrypoint()
def main_eval_pagekv_random_static():
    for dataset in DATASETS:
        run_eval.remote("pagekv_random_static", dataset)

@app.local_entrypoint()
def main_eval_tokenkv_quest_bounds_static():
    for dataset in DATASETS:
        run_eval.remote("tokenkv_quest_bounds_static", dataset)

@app.local_entrypoint()
def main_eval_tokenkv_snapkv_static():
    for dataset in DATASETS:
        run_eval.remote("tokenkv_snapkv_static", dataset)

@app.local_entrypoint()
def main_eval_tokenkv_h2o_static():
    for dataset in DATASETS:
        run_eval.remote("tokenkv_h2o_static", dataset)

@app.local_entrypoint()
def main_eval_tokenkv_recon_static():
    for dataset in DATASETS:
        run_eval.remote("tokenkv_recon_static", dataset)

@app.local_entrypoint()
def main_eval_tokenkv_expected_attention_static():
    for dataset in DATASETS:
        run_eval.remote("tokenkv_expected_attention_static", dataset)

@app.local_entrypoint()
def main_eval_tokenkv_random_static():
    for dataset in DATASETS:
        run_eval.remote("tokenkv_random_static", dataset)

@app.local_entrypoint()
def main_h2o():
    for dataset in DATASETS:
        run_inference.remote("h2o_static", dataset)

@app.local_entrypoint()
def main_h2o_static():
    for dataset in DATASETS:
        run_inference.remote("h2o_static", dataset)
