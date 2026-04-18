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

@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200
)
def run_baseline():
    import subprocess, shutil, os
    result = subprocess.run([
        "python", "pred_snap.py",
        "--model", "mistral-7B-instruct-v0.2",
        "--dataset", "qasper"
    ], cwd="/app/experiments/LongBench", capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # Save predictions to volume so they persist across runs
    os.makedirs("/models/predictions", exist_ok=True)
    for d in ["pred", "pred_e"]:
        src = f"/app/experiments/LongBench/{d}"
        if os.path.exists(src):
            shutil.copytree(src, f"/models/predictions/{d}", dirs_exist_ok=True)
            print(f"Saved {src} to volume")

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=300
)
def run_eval():
    import subprocess, shutil, os, json

    for d in ["pred", "pred_e"]:
        src = f"/models/predictions/{d}"
        dst = f"/app/experiments/LongBench/{d}"
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    os.makedirs("/app/experiments/LongBench/H2O/results/mistral-7B-instruct-v0.2", exist_ok=True)

    result = subprocess.run([
        "python", "eval.py",
        "--model", "mistral-7B-instruct-v0.2"
    ], cwd="/app/experiments/LongBench", capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # Print the result file
    result_path = "/app/experiments/LongBench/H2O/results/mistral-7B-instruct-v0.2/result.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            print("=== RESULTS ===")
            print(json.dumps(json.load(f), indent=2))
    else:
        print("Result file not found, listing directory:")
        for root, dirs, files in os.walk("/app/experiments/LongBench/H2O"):
            for file in files:
                print(os.path.join(root, file))

@app.local_entrypoint()
def main():
    run_baseline.remote()

@app.local_entrypoint()
def main_eval():
    run_eval.remote()