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
    gpu="A100",
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

    os.makedirs("/models/predictions", exist_ok=True)
    for d in ["pred", "pred_e"]:
        src = f"/app/experiments/LongBench/{d}"
        if os.path.exists(src):
            shutil.copytree(src, f"/models/predictions/{d}", dirs_exist_ok=True)
            print(f"Saved {src} to volume")

@app.function(
    gpu="A100",
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200
)
def run_snapkv():
    import subprocess, shutil, os
    result = subprocess.run([
        "python", "pred_snap.py",
        "--model", "mistral-7B-instruct-v0.2",
        "--dataset", "qasper",
        "--compress_args_path", "ablation_c4096_w32_k7_maxpool.json"
    ], cwd="/app/experiments/LongBench", capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    os.makedirs("/models/predictions_snapkv", exist_ok=True)
    for d in ["pred", "pred_e"]:
        src = f"/app/experiments/LongBench/{d}"
        if os.path.exists(src):
            shutil.copytree(src, f"/models/predictions_snapkv/{d}", dirs_exist_ok=True)
            print(f"Saved {src} to volume")

def _run_eval(pred_dir, model_name="mistral-7B-instruct-v0.2"):
    import subprocess, shutil, os, json

    for d in ["pred", "pred_e"]:
        src = f"/models/{pred_dir}/{d}"
        dst = f"/app/experiments/LongBench/{d}"
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    os.makedirs(f"/app/experiments/LongBench/H2O/results/{model_name}", exist_ok=True)

    result = subprocess.run([
        "python", "eval.py",
        "--model", model_name
    ], cwd="/app/experiments/LongBench", capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    result_path = f"/app/experiments/LongBench/H2O/results/{model_name}/result.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            print("=== RESULTS ===")
            print(json.dumps(json.load(f), indent=2))

@app.function(image=image, volumes={"/models": volume}, timeout=300)
def run_eval():
    _run_eval("predictions", "mistral-7B-instruct-v0.2")

@app.function(image=image, volumes={"/models": volume}, timeout=300)
def run_eval_snapkv():
    _run_eval("predictions_snapkv", "mistral-7B-instruct-v0.2ablation_c4096_w32_k7_maxpool")

@app.local_entrypoint()
def main():
    run_baseline.remote()

@app.local_entrypoint()
def main_list():
    list_volume.remote()

@app.local_entrypoint()
def main_eval():
    run_eval.remote()

@app.local_entrypoint()
def main_snapkv():
    run_snapkv.remote()

@app.local_entrypoint()
def main_eval_snapkv():
    run_eval_snapkv.remote()