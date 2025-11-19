# Training Quick Reference Card

## ⚡ Quick Start - Add New Language

```bash
# 1. Set language name
LANG="catalan"  # Change this!

# 2. Train tokenizer
python tools/tokenizer/train_bpe.py \
    --manifest data/${LANG}/train.jsonl \
    --output-prefix training/${LANG}/tokenizer/${LANG}_bpe \
    --vocab-size 12000

# 3. Fine-tune model
python trainers/train_gpt_v2.py \
    --train-manifest data/${LANG}/train_paired.jsonl \
    --val-manifest data/${LANG}/val_paired.jsonl \
    --tokenizer training/${LANG}/tokenizer/${LANG}_bpe.model \
    --base-checkpoint checkpoints/gpt.pth \
    --output-dir training/${LANG}/checkpoints \
    --epochs 10

# 4. Install for WebUI
python tools/install_trained_model.py \
    --checkpoint training/${LANG}/checkpoints/latest.pth \
    --tokenizer training/${LANG}/tokenizer/${LANG}_bpe.model \
    --output-name ${LANG}
```

## 📁 Directory Structure (DO NOT MODIFY checkpoints/)

```
✅ SAFE - Training directories:
training/${LANG}/
models/

❌ DANGEROUS - Base models (READ-ONLY):
checkpoints/
```

## 🎯 Model Naming Convention

| Type | Good ✅ | Bad ❌ |
|------|---------|--------|
| GPT | `gpt_catalan.pth` | `gpt.pth` |
| Tokenizer | `catalan_bpe.model` | `bpe.model` |
| Output dir | `training/catalan/` | `checkpoints/` |

## 🔧 Common Tasks

### Resume Training
```bash
python trainers/train_gpt_v2.py \
    --resume training/${LANG}/checkpoints/latest.pth \
    ...other args
```

### Use Specific GPU
```bash
CUDA_VISIBLE_DEVICES=1 python trainers/train_gpt_v2.py ...
```

### Multilingual Model
```bash
python trainers/train_gpt_v2.py \
    --train-manifest data/catalan/train.jsonl::ca \
    --train-manifest data/spanish/train.jsonl::es \
    --output-name multilingual_ca_es \
    ...
```

### Check Training Progress
```bash
# TensorBoard
tensorboard --logdir training/${LANG}/checkpoints/runs

# List checkpoints
ls -lh training/${LANG}/checkpoints/model_step*.pth
```

## ⚠️ Safety Checklist

Before running `train_gpt_v2.py`:

- [ ] `--output-dir` is NOT `checkpoints/`
- [ ] `--tokenizer` points to YOUR tokenizer
- [ ] `--base-checkpoint` is `checkpoints/gpt.pth` (not overwriting)
- [ ] Model name includes language (e.g., `gpt_catalan.pth`)

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "File exists" error | Change `--output-dir` or `--output-name` |
| Tokenizer not found in WebUI | Check naming: `{lang}_bpe.model` |
| Model not appearing | Run with `--model-dir models` |
| Base model overwritten | Restore from backup, use `--output-dir` |

## 📊 Hyperparameters

| Parameter | Small Dataset | Large Dataset |
|-----------|---------------|---------------|
| `--batch-size` | 2-4 | 8-16 |
| `--learning-rate` | 2e-5 | 1e-5 |
| `--epochs` | 10-20 | 5-10 |
| `--warmup-steps` | 500 | 1000 |
| `--grad-accumulation` | 2-4 | 1 |

## 🔗 Related Commands

```bash
# Prepare dataset
python tools/prepare_dataset.py \
    --audio-dir data/raw/${LANG}/ \
    --output data/${LANG}/train.jsonl

# Create prompt pairs
python tools/build_gpt_prompt_pairs.py \
    --manifest data/${LANG}/train.jsonl \
    --output data/${LANG}/train_paired.jsonl

# Test model
python -c "from indextts.infer_v2_modded import IndexTTS2; \
    tts = IndexTTS2(model_dir='models', gpt_checkpoint_path='models/gpt_${LANG}.pth'); \
    tts.infer('Test', 'prompt.wav', 'output.wav')"

# Launch WebUI
python webui.py --model-dir models
```

## 📚 Full Documentation

- Complete guide: `docs/TRAINING_GUIDE.md`
- Example script: `examples/train_new_language.sh`
- Tool reference: `tools/install_trained_model.py --help`
