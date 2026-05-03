import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import time
import torch
from snapkv.monkeypatch.monkeypatch import (
    replace_llama,
    replace_mistral,
    replace_mixtral,
    replace_llama_quest,
    replace_mistral_quest,
    replace_mixtral_quest,
    replace_llama_cluster,
    replace_mistral_cluster,
    replace_mixtral_cluster,
)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k",
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
        "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1", "llama-2-7B-32k-instruct", "mixtral-8x7B-instruct-v0.1","lwm-text-chat-1m", "lwm-text-1m"])
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--method', type=str, default='snapkv', choices=['snapkv', 'quest', 'clusterkv', 'heuristic_routing', 'xgb_routing'],
                        help="Compression method to enable when --compress_args_path is provided")
    parser.add_argument('--xgb_router_dir', type=str, default=None,
                        help="Directory containing xgb_router.json and xgb_router_metadata.json")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    parser.add_argument('--limit', type=int, default=None, help="Optional number of examples to run")
    parser.add_argument('--sample_offset', type=int, default=0, help="Optional starting index into the dataset")
    parser.add_argument('--write_model_name', type=str, default=None, help="Explicit output subdirectory name for pred/pred_e")
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name or "llama-2" in model_name or "lwm" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "mistral" in model_name or "mixtral" in model_name:
        prompt = prompt
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

HEURISTIC_ROUTING_CONFIGS = {
    "tokenkv_quest_bounds_dynamic100": {
        "method": "clusterkv",
        "compress_args_path": "tokenkv_quest_bounds_dynamic100_c4096_w64.json",
    },
    "tokenkv_quest_bounds_static": {
        "method": "clusterkv",
        "compress_args_path": "tokenkv_quest_bounds_c4096_w64.json",
    },
    "clusterattn_recon_static": {
        "method": "clusterkv",
        "compress_args_path": "clusterattn_recon_c4096_w64_p16.json",
    },
    "clusterattn_quest_bounds_static": {
        "method": "clusterkv",
        "compress_args_path": "clusterattn_quest_bounds_c4096_w64_p16.json",
    },
    "pagekv_quest_bounds_static": {
        "method": "clusterkv",
        "compress_args_path": "pagekv_quest_bounds_c4096_w64_p16.json",
    },
    "tokenkv_h2o_dynamic": {
        "method": "clusterkv",
        "compress_args_path": "tokenkv_h2o_dynamic_c4096_w64.json",
    },
}

CODE_MARKERS = (
    "def ", "class ", "import ", "from ", "return ", "self.", "->", "::",
    "();", "</", "#include", "public ", "private ",
)
SUMMARY_MARKERS = (
    "summary:", "summarize", "write a summary", "write a one-page summary",
)
QA_MARKERS = (
    "question:", "answer:", "passage", "given passages", "given a scientific article",
)

def choose_heuristic_route(prompt: str, max_gen: int):
    sample = (prompt[:512] + "\n" + prompt[-512:]).lower()
    code_hits = sum(marker in sample for marker in CODE_MARKERS)
    has_summary = any(marker in sample for marker in SUMMARY_MARKERS)
    has_qa = any(marker in sample for marker in QA_MARKERS)

    if max_gen >= 256 or has_summary:
        route = "tokenkv_quest_bounds_dynamic100"
        reason = "long_or_summary"
    elif code_hits >= 2:
        route = "clusterattn_recon_static"
        reason = "code_markers"
    elif has_qa and max_gen >= 128:
        route = "clusterattn_quest_bounds_static"
        reason = "qa_medium_long"
    elif has_qa:
        route = "pagekv_quest_bounds_static"
        reason = "qa_short"
    else:
        route = "clusterattn_recon_static"
        reason = "fallback"
    return route, reason

def prompt_features(prompt: str, max_gen: int, length=None):
    sample = (prompt[:512] + "\n" + prompt[-512:]).lower()
    return {
        "max_gen": max_gen,
        "prompt_chars": len(prompt),
        "sample_chars": len(sample),
        "code_marker_hits": sum(marker in sample for marker in CODE_MARKERS),
        "summary_marker_hits": sum(marker in sample for marker in SUMMARY_MARKERS),
        "qa_marker_hits": sum(marker in sample for marker in QA_MARKERS),
        "newline_ratio": prompt.count("\n") / max(len(prompt), 1),
        "digit_ratio": sum(ch.isdigit() for ch in prompt) / max(len(prompt), 1),
        "punct_ratio": sum((not ch.isalnum()) and (not ch.isspace()) for ch in prompt) / max(len(prompt), 1),
        "length": length or 0,
    }

def load_xgb_router(router_dir: str):
    from xgboost import XGBClassifier

    model_path = os.path.join(router_dir, "xgb_router.json")
    metadata_path = os.path.join(router_dir, "xgb_router_metadata.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing XGBoost router model: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing XGBoost router metadata: {metadata_path}")

    model = XGBClassifier()
    model.load_model(model_path)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

def choose_xgb_route(prompt: str, max_gen: int, length, dataset: str, xgb_router):
    model, metadata = xgb_router
    features = prompt_features(prompt, max_gen, length)
    row = []
    for name in metadata["feature_names"]:
        if name.startswith("dataset__"):
            row.append(1.0 if name.split("__", 1)[1] == dataset else 0.0)
        else:
            row.append(float(features.get(name, 0) or 0))
    pred_id = int(model.predict(np.array([row], dtype=np.float32))[0])
    label_names = metadata["label_names"]
    if pred_id < 0 or pred_id >= len(label_names):
        raise ValueError(f"XGBoost router predicted invalid label id {pred_id}")
    return label_names[pred_id], "xgb_classifier"

def apply_clusterkv_config(model, compress_args):
    layers = len(model.model.layers)
    for i in range(layers):
        cfg = model.model.layers[i].self_attn.config
        cfg.window_size = compress_args.get("window_size")
        cfg.max_capacity_prompt = compress_args.get("max_capacity_prompt")
        cfg.page_size = compress_args.get("page_size")
        cfg.update_policy = compress_args.get("update_policy")
        cfg.update_interval = compress_args.get("update_interval")
        cfg.n_clusters = compress_args.get("n_clusters")
        cfg.ranking_backend = compress_args.get("ranking_backend")
        cfg.observation_window = compress_args.get("observation_window")
        cfg.selection_granularity = compress_args.get("selection_granularity")
        cfg.clustering_backend = compress_args.get("clustering_backend", "kmeanspp")
        cfg.num_block = compress_args.get("num_block", 12)
        cfg.theta = compress_args.get("theta", 0.0)
        cfg.n_future_positions = compress_args.get("n_future_positions", 512)
        cfg.n_sink = compress_args.get("n_sink", 4)
        cfg.use_covariance = compress_args.get("use_covariance", True)
        cfg.use_vnorm = compress_args.get("use_vnorm", True)
        cfg.epsilon = compress_args.get("epsilon", 0.0)
        cfg.hidden_states_buffer_size = compress_args.get("hidden_states_buffer_size", 128)

def reset_kv_runtime_state(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.kv_seq_len = 0
        if hasattr(attn, "kv_cluster"):
            attn.kv_cluster.reset()

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen,
                        prompt_format, dataset, model_name,
                        model2path, out_path,
                        compress=False,
                        method='snapkv',
                        window_sizes=None,
                        max_capacity_prompts=None,
                        kernel_sizes=None,
                        pooling=None,
                        window_size=None,
                        max_capacity_prompt=None,
                        page_size=None,
                        update_policy=None,
                        update_interval=None,
                        n_clusters=None,
                        ranking_backend=None,
                        observation_window=None,
                        selection_granularity=None,
                        clustering_backend=None,
                        num_block=None,
                        theta=None,
                        routing_configs=None,
                        xgb_router=None):
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device="cuda", compress=compress)
    device = model.device

    # Reset memory stats after model load so we only measure inference
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    memory_after_load_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"\n=== MEMORY: after model load: {memory_after_load_mb:.1f} MB ===")

    printed = False
    context_lengths = []
    latencies_s = []
    prefill_latencies_s = []
    decode_latencies_s = []
    generate_peak_mbs = []
    generated_tokens = []
    routing_counts = {}
    profiled_flops = None
    profiled_latency_s = None

    for idx, json_obj in enumerate(tqdm(data)):
        route = None
        route_reason = None
        if compress:
            layers = len(model.model.layers)
            if method == 'heuristic_routing':
                pass
            elif method == 'snapkv':
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                for i in range(layers):
                    model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                    model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                    model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                    model.model.layers[i].self_attn.config.pooling = pooling
            elif method in ('quest', 'clusterkv'):
                for i in range(layers):
                    cfg = model.model.layers[i].self_attn.config
                    cfg.window_size = window_size
                    cfg.max_capacity_prompt = max_capacity_prompt
                    cfg.page_size = page_size
                    cfg.update_policy = update_policy
                    cfg.update_interval = update_interval
                    cfg.n_clusters = n_clusters
                    if method == 'clusterkv':
                        cfg.ranking_backend = ranking_backend
                        cfg.observation_window = observation_window
                        cfg.selection_granularity = selection_granularity
                        cfg.clustering_backend = clustering_backend
                        if num_block is not None:
                            cfg.num_block = num_block
                        if theta is not None:
                            cfg.theta = theta
            else:
                raise ValueError(f"Compression method {method} not supported")

        prompt = prompt_format.format(**json_obj)
        if compress and method in ('heuristic_routing', 'xgb_routing'):
            if method == 'heuristic_routing':
                route, route_reason = choose_heuristic_route(prompt, max_gen)
            else:
                route, route_reason = choose_xgb_route(
                    prompt,
                    max_gen,
                    json_obj.get("length"),
                    dataset,
                    xgb_router,
                )
            route_cfg = routing_configs[route]
            route_compress_args = route_cfg["compress_args"]
            apply_clusterkv_config(model, route_compress_args)
            routing_counts[route] = routing_counts.get(route, 0) + 1
            if idx < 5:
                print(
                    f"{method.upper()}_ROUTE "
                    f"example={idx} route={route} reason={route_reason} max_gen={max_gen} "
                    f"capacity={route_compress_args.get('max_capacity_prompt')} "
                    f"window={route_compress_args.get('window_size')} "
                    f"granularity={route_compress_args.get('selection_granularity')} "
                    f"backend={route_compress_args.get('ranking_backend')} "
                    f"policy={route_compress_args.get('update_policy')} "
                    f"interval={route_compress_args.get('update_interval')}"
                )

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        context_lengths.append(context_length)

        if not printed:
            print(prompt)
            printed = True

        generate_kwargs = dict(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length + 1,
        )
        if dataset == "samsum":
            generate_kwargs["eos_token_id"] = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]

        # Measure prefill latency (processing the input prompt)
        torch.cuda.synchronize()
        reset_kv_runtime_state(model)
        prefill_start = time.perf_counter()
        with torch.no_grad():
            prefill_output = model(input.input_ids, attention_mask=input.get("attention_mask"))
        torch.cuda.synchronize()
        prefill_latency_s = time.perf_counter() - prefill_start
        prefill_latencies_s.append(prefill_latency_s)
        # The diagnostic prefill returns a cache-bearing output object. Drop it
        # before measuring generate() peak memory, otherwise the diagnostic KV
        # cache is counted together with the real generation KV cache.
        del prefill_output
        
        # Measure total latency (prefill + decode)
        torch.cuda.synchronize()
        reset_kv_runtime_state(model)
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        if idx == 0:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_flops=True,
            ) as prof:
                output = model.generate(**generate_kwargs)[0]
            torch.cuda.synchronize()
            profiled_latency_s = time.perf_counter() - start_time
            profiled_flops = 0.0
            for evt in prof.key_averages():
                flops = getattr(evt, "flops", 0) or 0
                profiled_flops += float(flops)
        else:
            output = model.generate(**generate_kwargs)[0]
            torch.cuda.synchronize()

        step_latency_s = time.perf_counter() - start_time
        generate_peak_mbs.append(torch.cuda.max_memory_allocated() / 1024**2)
        latencies_s.append(step_latency_s)
        
        # Calculate decode latency as total - prefill
        decode_latency_s = max(0, step_latency_s - prefill_latency_s)
        decode_latencies_s.append(decode_latency_s)

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        generated_token_count = max(int(output.shape[0] - context_length), 0)
        generated_tokens.append(generated_token_count)
        with open(out_path, "a", encoding="utf-8") as f:
            record = {
                "example_idx": idx,
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
                "context_length": context_length,
                "latency_s": step_latency_s,
                "prefill_latency_s": prefill_latency_s,
                "decode_latency_s": decode_latency_s,
                "generated_tokens": generated_token_count,
            }
            if route is not None:
                record["route"] = route
                record["route_reason"] = route_reason
                if method == "heuristic_routing":
                    record["heuristic_route"] = route
                    record["heuristic_route_reason"] = route_reason
                elif method == "xgb_routing":
                    record["xgb_route"] = route
                    record["xgb_route_reason"] = route_reason
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    # Memory summary after all inference
    torch.cuda.synchronize()
    peak_mb = max(generate_peak_mbs) if generate_peak_mbs else torch.cuda.max_memory_allocated() / 1024**2
    peak_gb = peak_mb / 1024
    current_mb = torch.cuda.memory_allocated() / 1024**2
    avg_ctx = sum(context_lengths) / len(context_lengths) if context_lengths else 0
    max_ctx = max(context_lengths) if context_lengths else 0
    total_latency_s = sum(latencies_s)
    avg_latency_s = total_latency_s / len(latencies_s) if latencies_s else 0
    max_latency_s = max(latencies_s) if latencies_s else 0
    
    # Calculate prefill and decode latencies
    total_prefill_latency_s = sum(prefill_latencies_s)
    avg_prefill_latency_s = total_prefill_latency_s / len(prefill_latencies_s) if prefill_latencies_s else 0
    max_prefill_latency_s = max(prefill_latencies_s) if prefill_latencies_s else 0
    
    total_decode_latency_s = sum(decode_latencies_s)
    avg_decode_latency_s = total_decode_latency_s / len(decode_latencies_s) if decode_latencies_s else 0
    max_decode_latency_s = max(decode_latencies_s) if decode_latencies_s else 0
    
    avg_generated_tokens = sum(generated_tokens) / len(generated_tokens) if generated_tokens else 0
    tokens_per_second = (sum(generated_tokens) / total_latency_s) if total_latency_s > 0 else 0
    profiled_tflops = (profiled_flops / 1e12) if profiled_flops is not None else None
    profiled_tflops_per_s = (
        profiled_tflops / profiled_latency_s
        if profiled_tflops is not None and profiled_latency_s and profiled_latency_s > 0
        else None
    )

    print(f"\n{'='*50}")
    print(f"=== INFERENCE SUMMARY ({method if compress else 'full'}) ===")
    print(f"Peak GPU memory allocated:    {peak_mb:.1f} MB  ({peak_gb:.2f} GB)")
    print(f"Peak GPU source:              generate() only")
    print(f"Current GPU memory allocated: {current_mb:.1f} MB")
    print(f"Memory for KV cache (approx): {peak_mb - memory_after_load_mb:.1f} MB")
    print(f"Average context length:       {avg_ctx:.0f} tokens")
    print(f"Max context length:           {max_ctx} tokens")
    print(f"Total latency:                {total_latency_s:.3f} s")
    print(f"Average prefill latency:      {avg_prefill_latency_s:.3f} s")
    print(f"Average decode latency:       {avg_decode_latency_s:.3f} s")
    print(f"Max prefill latency:          {max_prefill_latency_s:.3f} s")
    print(f"Max decode latency:           {max_decode_latency_s:.3f} s")
    print(f"Average latency / example:    {avg_latency_s:.3f} s")
    print(f"Max latency / example:        {max_latency_s:.3f} s")
    print(f"Average generated tokens:     {avg_generated_tokens:.1f}")
    print(f"Generation throughput:        {tokens_per_second:.2f} tok/s")
    if profiled_flops is not None:
        print(f"Profiled FLOPs (1st example): {profiled_flops:.0f}")
    if profiled_tflops is not None:
        print(f"Profiled TFLOPs (1st example): {profiled_tflops:.6f}")
    if profiled_tflops_per_s is not None:
        print(f"Profiled TFLOPs/s (1st example): {profiled_tflops_per_s:.6f}")
    if routing_counts:
        print(f"Route counts:                  {json.dumps(routing_counts, sort_keys=True)}")
    print(f"{'='*50}\n")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, compress=False):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map="auto", use_cache=True, use_flash_attention_2=True)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    elif "llama-2" in model_name or "lwm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map="auto", use_cache=True, use_flash_attention_2=True)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    elif "mistral" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map="auto", use_cache=True, use_flash_attention_2=True)
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right", use_fast=False)
    elif "mixtral" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map="auto", use_cache=True, use_flash_attention_2=True)
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        raise ValueError(f"Model {model_name} not supported!")
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    max_length = model2maxlen[model_name]

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    if args.dataset not in datasets:
        raise ValueError(f"Dataset {args.dataset} not found in datasets")

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    dataset = args.dataset
    routing_configs = None
    xgb_router = None
    if args.method in ("heuristic_routing", "xgb_routing"):
        compress = True
        compress_args = {}
        write_model_name = args.write_model_name or (model_name + args.method)
        routing_configs = {}
        for route_name, route_spec in HEURISTIC_ROUTING_CONFIGS.items():
            with open(os.path.join("config", route_spec["compress_args_path"]), "r") as f:
                routing_configs[route_name] = {
                    "method": route_spec["method"],
                    "compress_args": json.load(f),
                }
        if args.method == "xgb_routing":
            if not args.xgb_router_dir:
                raise ValueError("--xgb_router_dir is required for --method xgb_routing")
            xgb_router = load_xgb_router(args.xgb_router_dir)
        replace_llama_cluster()
        replace_mistral_cluster()
        replace_mixtral_cluster()
    elif args.compress_args_path:
        compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
        compress = True
        write_model_name = args.write_model_name or (model_name + args.compress_args_path.split(".")[0])
        if args.method == 'snapkv':
            replace_llama()
            replace_mistral()
            replace_mixtral()
        elif args.method == 'quest':
            replace_llama_quest()
            replace_mistral_quest()
            replace_mixtral_quest()
        elif args.method == 'clusterkv':
            replace_llama_cluster()
            replace_mistral_cluster()
            replace_mixtral_cluster()
        else:
            raise ValueError(f"Compression method {args.method} not supported")
    else:
        compress = False
        compress_args = None
        write_model_name = args.write_model_name or model_name

    if args.e:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', trust_remote_code=True)
        if not os.path.exists(f"pred_e/{write_model_name}"):
            os.makedirs(f"pred_e/{write_model_name}")
        out_path = f"pred_e/{write_model_name}/{dataset}.jsonl"
    else:
        data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
        if not os.path.exists(f"pred_e/{write_model_name}"):
            os.makedirs(f"pred_e/{write_model_name}")
        out_path = f"pred_e/{write_model_name}/{dataset}.jsonl"

    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    data_all = [data_sample for data_sample in data]
    if args.sample_offset or args.limit is not None:
        start = max(args.sample_offset, 0)
        end = None if args.limit is None else start + max(args.limit, 0)
        data_all = data_all[start:end]

    if compress_args is not None:
        get_pred_single_gpu(
            data_all, max_length, max_gen, prompt_format, dataset, model_name,
            model2path, out_path, compress, args.method,
            routing_configs=routing_configs,
            xgb_router=xgb_router,
            **compress_args
        )
    else:
        get_pred_single_gpu(
            data_all, max_length, max_gen, prompt_format, dataset, model_name,
            model2path, out_path, compress, args.method
        )
