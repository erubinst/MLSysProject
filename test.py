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
    "snapkv": {
        "script": "pred_snap.py",
        "extra_args": ["--compress_args_path", "ablation_c4096_w32_k7_maxpool.json"],
        "model_name": "mistral-7B-instruct-v0.2ablation_c4096_w32_k7_maxpool",
    },
    "h2o": {
        "script": "pred_h2o.py",
        "extra_args": ["--max_capacity_prompt", "2048", "--window_size", "32"],
        "model_name": "mistral-7B-instruct-v0.2-h2o-budget2048",
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
    import subprocess, shutil, os

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
    memory = {
        "baseline": {"peak_gb": 23.20, "kv_cache_mb": 9429},
        "snapkv":   {"peak_gb": 20.89, "kv_cache_mb": 7072},
        "h2o":      {"peak_gb": 19.53, "kv_cache_mb": 5672},
        "ack":      {"peak_gb": None,  "kv_cache_mb": None},
    }

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        method = data["method"]
        dataset = data["dataset"]
        score = data["score"]
        if method not in rows:
            rows[method] = {}
        rows[method][dataset] = score

    if not rows:
        print("No results found yet.")
        return

    # Build CSV
    all_datasets = sorted({d for m in rows.values() for d in m})
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Method"] + all_datasets + ["Average", "Peak GPU (GB)", "KV Cache (MB)"])

    for method in ["baseline", "snapkv", "h2o", "ack"]:
        if method not in rows:
            continue
        scores = [rows[method].get(d) for d in all_datasets]
        valid_scores = [s for s in scores if s is not None]
        avg = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None
        mem = memory.get(method, {})
        row = (
            [method]
            + [round(s, 2) if s is not None else "" for s in scores]
            + [avg if avg is not None else ""]
            + [mem.get("peak_gb") or ""]
            + [mem.get("kv_cache_mb") or ""]
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
    methods_to_run = ["baseline", "snapkv", "h2o"]
    for method in methods_to_run:
        for dataset in DATASETS:
            print(f"\nSubmitting {method} / {dataset}...")
            run_inference.remote(method, dataset)


@app.local_entrypoint()
def main_eval_all():
    """Score all completed inference runs and generate CSV."""
    methods_to_run = ["baseline", "snapkv", "h2o"]
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
        run_inference.remote("snapkv", dataset)

@app.local_entrypoint()
def main_h2o():
    for dataset in DATASETS:
        run_inference.remote("h2o", dataset)