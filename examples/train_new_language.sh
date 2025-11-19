#!/bin/bash
# Example training script for adding a new language to IndexTTS-2
# This script demonstrates best practices for training without overwriting base models

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Language/experiment name (change this!)
LANGUAGE="catalan"  # Could be: french, spanish, portuguese, etc.

# Data paths (adjust to your dataset)
TRAIN_MANIFEST="data/${LANGUAGE}_dataset/train_paired.jsonl"
VAL_MANIFEST="data/${LANGUAGE}_dataset/val_paired.jsonl"
RAW_TEXT_MANIFEST="data/${LANGUAGE}_dataset/train.jsonl"  # For tokenizer training

# Training configuration
VOCAB_SIZE=12000
BATCH_SIZE=4
EPOCHS=10
LEARNING_RATE=2e-5
GPU_ID=0  # Change if using different GPU

# Directory structure
TRAINING_ROOT="training/${LANGUAGE}"
TOKENIZER_DIR="${TRAINING_ROOT}/tokenizer"
CHECKPOINT_DIR="${TRAINING_ROOT}/checkpoints"
FINAL_MODELS_DIR="models"

# Base model to fine-tune from
BASE_CHECKPOINT="checkpoints/gpt.pth"
BASE_CONFIG="checkpoints/config.yaml"

# ============================================================================
# SAFETY CHECKS
# ============================================================================

echo "=========================================="
echo "IndexTTS-2 Training Script"
echo "=========================================="
echo "Language: ${LANGUAGE}"
echo "Training directory: ${TRAINING_ROOT}"
echo "Final models will be saved to: ${FINAL_MODELS_DIR}"
echo ""

# Check if trying to write to protected directory
if [[ "${CHECKPOINT_DIR}" == "checkpoints"* ]]; then
    echo "❌ ERROR: Output directory cannot be inside 'checkpoints/'"
    echo "This would overwrite base models!"
    exit 1
fi

# Warn if base models might be overwritten
if [ -d "checkpoints" ]; then
    if [ -f "checkpoints/gpt.pth" ]; then
        echo "✓ Base model found: checkpoints/gpt.pth"
    else
        echo "⚠️  WARNING: Base checkpoint not found at checkpoints/gpt.pth"
        read -p "Continue anyway? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check if data exists
if [ ! -f "${RAW_TEXT_MANIFEST}" ]; then
    echo "❌ ERROR: Training data not found at ${RAW_TEXT_MANIFEST}"
    echo "Please prepare your dataset first using tools/prepare_dataset.py"
    exit 1
fi

echo ""
read -p "Ready to start training? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ============================================================================
# STEP 1: Train BPE Tokenizer
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 1: Training BPE Tokenizer"
echo "=========================================="

mkdir -p "${TOKENIZER_DIR}"

TOKENIZER_PREFIX="${TOKENIZER_DIR}/${LANGUAGE}_bpe"
TOKENIZER_MODEL="${TOKENIZER_PREFIX}.model"

if [ -f "${TOKENIZER_MODEL}" ]; then
    echo "⚠️  Tokenizer already exists at ${TOKENIZER_MODEL}"
    read -p "Skip tokenizer training? [Y/n]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Skipping tokenizer training..."
    else
        echo "Training new tokenizer..."
        python tools/tokenizer/train_bpe.py \
            --manifest "${RAW_TEXT_MANIFEST}" \
            --output-prefix "${TOKENIZER_PREFIX}" \
            --vocab-size "${VOCAB_SIZE}" \
            --character-coverage 0.9995
    fi
else
    echo "Training tokenizer with ${VOCAB_SIZE} vocabulary size..."
    python tools/tokenizer/train_bpe.py \
        --manifest "${RAW_TEXT_MANIFEST}" \
        --output-prefix "${TOKENIZER_PREFIX}" \
        --vocab-size "${VOCAB_SIZE}" \
        --character-coverage 0.9995
fi

if [ ! -f "${TOKENIZER_MODEL}" ]; then
    echo "❌ ERROR: Tokenizer training failed!"
    exit 1
fi

echo "✓ Tokenizer saved to: ${TOKENIZER_MODEL}"

# ============================================================================
# STEP 2: Fine-tune GPT Model
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 2: Fine-tuning GPT Model"
echo "=========================================="

mkdir -p "${CHECKPOINT_DIR}"

# Check if paired manifests exist
if [ ! -f "${TRAIN_MANIFEST}" ]; then
    echo "❌ ERROR: Paired training manifest not found: ${TRAIN_MANIFEST}"
    echo "Please run tools/build_gpt_prompt_pairs.py first"
    exit 1
fi

if [ ! -f "${VAL_MANIFEST}" ]; then
    echo "⚠️  WARNING: Validation manifest not found: ${VAL_MANIFEST}"
    echo "Using training manifest for validation (not recommended)"
    VAL_MANIFEST="${TRAIN_MANIFEST}"
fi

echo "Configuration:"
echo "  Base checkpoint: ${BASE_CHECKPOINT}"
echo "  Tokenizer: ${TOKENIZER_MODEL}"
echo "  Output directory: ${CHECKPOINT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  GPU: ${GPU_ID}"
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Train!
python trainers/train_gpt_v2.py \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --tokenizer "${TOKENIZER_MODEL}" \
    --config "${BASE_CONFIG}" \
    --base-checkpoint "${BASE_CHECKPOINT}" \
    --output-dir "${CHECKPOINT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --amp \
    --grad-accumulation 1 \
    --warmup-steps 500 \
    --log-interval 50 \
    --val-interval 500

echo ""
echo "✓ Training completed!"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"

# ============================================================================
# STEP 3: Select Best Checkpoint
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 3: Select Best Checkpoint"
echo "=========================================="

# List available checkpoints
echo "Available checkpoints:"
ls -lh "${CHECKPOINT_DIR}"/model_step*.pth 2>/dev/null || true

echo ""
echo "Which checkpoint do you want to use? (e.g., 5000 for model_step5000.pth)"
echo "Or press Enter to use latest.pth"
read -p "Step number: " STEP_NUMBER

if [ -z "${STEP_NUMBER}" ]; then
    BEST_CHECKPOINT="${CHECKPOINT_DIR}/latest.pth"
else
    BEST_CHECKPOINT="${CHECKPOINT_DIR}/model_step${STEP_NUMBER}.pth"
fi

if [ ! -f "${BEST_CHECKPOINT}" ]; then
    echo "❌ ERROR: Checkpoint not found: ${BEST_CHECKPOINT}"
    exit 1
fi

echo "Selected checkpoint: ${BEST_CHECKPOINT}"

# ============================================================================
# STEP 4: Install Model for WebUI
# ============================================================================

echo ""
echo "=========================================="
echo "STEP 4: Install Model for WebUI Use"
echo "=========================================="

echo "Installing model as: ${LANGUAGE}"
echo "  Checkpoint: ${BEST_CHECKPOINT}"
echo "  Tokenizer: ${TOKENIZER_MODEL}"
echo "  Output directory: ${FINAL_MODELS_DIR}"
echo ""

python tools/install_trained_model.py \
    --checkpoint "${BEST_CHECKPOINT}" \
    --tokenizer "${TOKENIZER_MODEL}" \
    --output-name "${LANGUAGE}" \
    --output-dir "${FINAL_MODELS_DIR}" \
    --description "${LANGUAGE} fine-tuned IndexTTS-2 model"

# ============================================================================
# DONE!
# ============================================================================

echo ""
echo "=========================================="
echo "✅ TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Your ${LANGUAGE} model is now ready to use!"
echo ""
echo "To use in WebUI:"
echo "  python webui.py --model-dir ${FINAL_MODELS_DIR}"
echo ""
echo "The model will appear in the dropdown as:"
echo "  gpt_${LANGUAGE}.pth"
echo ""
echo "Tokenizer auto-detected as:"
echo "  ${LANGUAGE}_bpe.model"
echo ""
echo "Training files preserved in:"
echo "  ${TRAINING_ROOT}/"
echo ""
echo "Base models remain untouched in:"
echo "  checkpoints/"
echo ""
echo "=========================================="
