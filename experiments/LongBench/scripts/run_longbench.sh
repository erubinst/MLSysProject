# SnapKV example
CUDA_VISIBLE_DEVICES=6 python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w32_k7_maxpool.json

# Quest-style page retrieval example
# CUDA_VISIBLE_DEVICES=6 python pred_snap.py --model mistral-7B-instruct-v0.2 --method quest --compress_args_path quest_c4096_w64_p16.json
