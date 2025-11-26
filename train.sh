#!/bin/bash
# Training script for Catalan TTS model (festcat + lafrescat datasets)
# Combined dataset: 29,088 training pairs, 1,498 validation pairs
# Tokenizer: Catalan BPE 24k vocabulary
# GPU: Blackwell (102GB VRAM) - PyTorch GPU 1

# Force PyTorch to use ONLY the Blackwell GPU
export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1  # Disabled for better performance

uv run python trainers/train_gpt_v2.py \
    --train-manifest processed_catalan/festcat_processed/gpt_pairs_train.jsonl::ca \
    --train-manifest processed_catalan/lafrescat_processed/gpt_pairs_train.jsonl::ca \
    --val-manifest processed_catalan/festcat_processed/gpt_pairs_val.jsonl::ca \
    --val-manifest processed_catalan/lafrescat_processed/gpt_pairs_val.jsonl::ca \
    --tokenizer checkpoints/catalan_bpe_24k.model \
    --config checkpoints/config.yaml \
    --base-checkpoint checkpoints/gpt_catalan_24k.pth \
    --output-dir trained_ckpts_catalan \
    --batch-size 40 \
    --grad-accumulation 1 \
    --epochs 10 \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --warmup-steps 1000 \
    --log-interval 100 \
    --val-interval 2000 \
    --grad-clip 1.0 \
    --text-loss-weight 0.2 \
    --mel-loss-weight 0.8 \
    --amp \
    --resume auto
