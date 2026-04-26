import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
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
    parser.add_argument('--method', type=str, default='snapkv', choices=['snapkv', 'quest', 'clusterkv'],
                        help="Compression method to enable when --compress_args_path is provided")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    parser.add_argument('--limit', type=int, default=None, help="Optional number of examples to run")
    parser.add_argument('--sample_offset', type=int, default=0, help="Optional starting index into the dataset")
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
                        clustering_backend=None):
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device="cuda", compress=compress)
    device = model.device

    # Reset memory stats after model load so we only measure inference
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    memory_after_load_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"\n=== MEMORY: after model load: {memory_after_load_mb:.1f} MB ===")

    printed = False
    context_lengths = []

    for json_obj in tqdm(data):
        if compress:
            layers = len(model.model.layers)
            if method == 'snapkv':
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
            else:
                raise ValueError(f"Compression method {method} not supported")

        prompt = prompt_format.format(**json_obj)
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

        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
            )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    # Memory summary after all inference
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_gb = peak_mb / 1024
    current_mb = torch.cuda.memory_allocated() / 1024**2
    avg_ctx = sum(context_lengths) / len(context_lengths) if context_lengths else 0
    max_ctx = max(context_lengths) if context_lengths else 0

    print(f"\n{'='*50}")
    print(f"=== MEMORY SUMMARY ({method if compress else 'full'}) ===")
    print(f"Peak GPU memory allocated:    {peak_mb:.1f} MB  ({peak_gb:.2f} GB)")
    print(f"Current GPU memory allocated: {current_mb:.1f} MB")
    print(f"Memory for KV cache (approx): {peak_mb - memory_after_load_mb:.1f} MB")
    print(f"Average context length:       {avg_ctx:.0f} tokens")
    print(f"Max context length:           {max_ctx} tokens")
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
    if args.compress_args_path:
        compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
        compress = True
        write_model_name = model_name + args.compress_args_path.split(".")[0]
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
        write_model_name = model_name

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
            model2path, out_path, compress, args.method, **compress_args
        )
    else:
        get_pred_single_gpu(
            data_all, max_length, max_gen, prompt_format, dataset, model_name,
            model2path, out_path, compress, args.method
        )
