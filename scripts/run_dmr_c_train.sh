#!/usr/bin/env bash
# Lever C: LoRA-tune the reader to compose V-MQAR answers from page-tagged document memory,
# on the DISJOINT aux pool (scripts/dmr_build_aux_sft_memory.py output). Protocol-clean.
# Usage: bash scripts/run_dmr_c_train.sh <2b|4b> [gpu_id]
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${1:?usage: run_dmr_c_train.sh <2b|4b> [gpu]}"
GPU="${2:-0}"
case "$SIZE" in
  2b) MODEL=Qwen/Qwen3-VL-2B-Instruct ;;
  4b) MODEL=Qwen/Qwen3-VL-4B-Instruct ;;
  *) echo "size must be 2b|4b"; exit 1 ;;
esac
DATA=code/runs/20260630/dmr_sft_aux_vmqar
OUT=code/runs/20260630/dmr_sft_c_${SIZE}
mkdir -p "$OUT"
echo "[C-train $SIZE] model=$MODEL gpu=$GPU out=$OUT"
CUDA_VISIBLE_DEVICES="$GPU" MODEL_PATH="$MODEL" PYTHONPATH=src \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup .venv/bin/python -m eccv26.train_sft \
  "model.path=${MODEL}" "model.dtype=bf16" "model.device_map=null" \
  "dma.enable=false" "cross_attn.enable=false" \
  "data.train_jsonl=${DATA}/train.jsonl" "data.val_jsonl=${DATA}/val.jsonl" \
  "data.image_root=." "data.max_images=4" "data.image_max_pixels=262144" \
  "peft.enable=true" "peft.lora_r=16" "peft.lora_alpha=32" \
  "train.per_device_batch_size=1" "train.grad_accum_steps=8" \
  "train.lr=1.0e-4" "train.max_steps=1000" "train.warmup_steps=50" \
  "output.dir=${OUT}" "output.save_every=250" "output.save_best_on_val=true" \
  > "$OUT/train.log" 2>&1 &
echo "  pid=$! -> $OUT/train.log"
