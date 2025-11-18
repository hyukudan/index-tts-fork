# IndexTTS-2 Checkpoints

This directory contains the official IndexTTS-2 model checkpoints downloaded from Hugging Face.

## Download Instructions

The model checkpoints (4.4 GB) are not included in the git repository. Download them using:

```bash
# Using huggingface-cli (deprecated but works)
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints

# Or using the newer hf command
hf download IndexTeam/IndexTTS-2 --local-dir checkpoints
```

## Model Files

After downloading, you should have:

- `config.yaml` (3 KB) - Model configuration
- `gpt.pth` (3.3 GB) - GPT model weights
- `s2mel.pth` (1.2 GB) - Spectrogram-to-mel model weights
- `bpe.model` (465 KB) - Byte-pair encoding model
- `feat1.pt`, `feat2.pt` - Feature extractors
- `wav2vec2bert_stats.pt` - Audio statistics
- `qwen0.6bemo4-merge/` - Additional model files
- `README.md` - Official model documentation

## Verified Configuration

✅ Tested with:
- PyTorch 2.9.1+cu128
- CUDA 12.8
- Python 3.12.3
- NVIDIA Blackwell RTX PRO 6000

## Source

Official Hugging Face repository: https://huggingface.co/IndexTeam/IndexTTS-2
