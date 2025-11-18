# GPU Troubleshooting Guide - IndexTTS

## Common Issues with Modern GPUs (Blackwell, Ada, Ampere)

### Problem: Errors with Wav2Vec2Bert and Transformer Models

**Symptoms:**
- Crashes during model loading
- NaN values in outputs
- CUDA errors related to Flash Attention
- "RuntimeError: CUDA error" during inference
- Slow performance with transformer models

**Root Causes:**

1. **Flash Attention Compatibility**
   - The hardcoded `flash-attn` version in `pyproject.toml` may not support Blackwell (compute capability 10.0)
   - Solution: Install Flash Attention from source or use a compatible wheel

2. **FP16 vs BF16 Precision**
   - Blackwell GPUs have native BF16 support which is more stable than FP16
   - FP16 can cause numerical instability (NaN/overflow) in Wav2Vec2Bert on Blackwell
   - Solution: Consider using BF16 if supported by the model

3. **Transformer Model Loading**
   - Models loaded without proper `attn_implementation` parameter
   - Location: `indextts/infer_v2_modded.py:217-220`
   - The `SeamlessM4TFeatureExtractor` and `Wav2Vec2BertModel` load without dtype specification

4. **CUDA Version Mismatch**
   - PyTorch 2.8.0 requires CUDA 12.8
   - Older drivers may not support CUDA 12.8 features
   - Check: `nvidia-smi` should show driver version 560+ for CUDA 12.8

### Solutions

#### 1. Check Your Environment
```bash
python webui.py --verbose
# or
python webui_parallel.py --verbose
```

Look for the startup diagnostics:
- GPU name and compute capability
- CUDA version
- Flash Attention availability
- Memory information
- Suggested worker count

#### 2. Flash Attention Issues

**If you see: "⚠️ Flash Attention not found"**

Option A - Install from PyPI (if available for your GPU):
```bash
pip install flash-attn --no-build-isolation
```

Option B - Build from source (for Blackwell/newest GPUs):
```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
MAX_JOBS=4 python setup.py install
```

Option C - Disable Flash Attention in transformers:
```python
# In indextts/utils/maskgct_utils.py
semantic_model = Wav2Vec2BertModel.from_pretrained(
    "facebook/w2v-bert-2.0",
    attn_implementation="eager"  # Use PyTorch's native attention
)
```

#### 3. FP16 Issues on Blackwell

**If you see NaN values or numerical instability:**

Option A - Use CPU precision (slower but stable):
```bash
python webui.py  # Don't use --is_fp16
```

Option B - Modify model loading to use BF16:
```python
# In indextts/infer_v2_modded.py around line 220
self.semantic_model = self.semantic_model.to(self.device).to(torch.bfloat16)
```

#### 4. Out of Memory Errors

The improved error handling will now:
- Show clear OOM messages
- Automatically clear CUDA cache
- Suggest reducing parameters

**Manual fixes:**
```bash
# Reduce max_mel_tokens in the UI (default 1500 → try 1000)
# Reduce max_text_tokens_per_sentence (default 120 → try 80)
# Reduce parallel workers in webui_parallel.py
```

#### 5. torch.load Warnings

**If you see: "FutureWarning: weights_only=False"**

This is safe for now but should be fixed. The warnings come from:
- `indextts/infer_v2_modded.py:256` - campplus model
- `indextts/infer_v2_modded.py:280, 284` - emotion/speaker matrices

To fix:
```python
# Change from:
torch.load(path, map_location="cpu")

# To:
torch.load(path, map_location="cpu", weights_only=True)
```

### GPU-Specific Recommendations

#### RTX 6000 Blackwell (Compute Capability 10.0)
- ✅ Use BF16 instead of FP16 for better stability
- ✅ Ensure Flash Attention is built for sm_100
- ✅ Update to NVIDIA driver 560+
- ✅ Use CUDA 12.8
- 💡 Start with 2-4 parallel workers (48GB VRAM)

#### RTX 4090 (Compute Capability 8.9)
- ✅ FP16 works well
- ✅ Flash Attention readily available
- 💡 Start with 2-3 parallel workers (24GB VRAM)

#### RTX 3090 (Compute Capability 8.6)
- ✅ FP16 works well
- ✅ Flash Attention supported
- 💡 Start with 2 parallel workers (24GB VRAM)

### Optimization Tips

1. **Worker Count** (for webui_parallel.py):
   - Formula: `workers = GPU_VRAM_GB / 8`
   - RTX 6000 (48GB): 4-6 workers
   - RTX 4090 (24GB): 2-3 workers
   - Monitor memory with: `watch -n 1 nvidia-smi`

2. **Batch Size**:
   - Start small and increase gradually
   - Watch for OOM errors in the UI
   - The new error handling will warn you

3. **CUDA Settings** (automatically set for Blackwell):
   - `CUDA_LAUNCH_BLOCKING=0` - async execution
   - `TORCH_CUDNN_V8_API_ENABLED=1` - cuDNN v8 optimizations

### Debugging Commands

```bash
# Check GPU info
nvidia-smi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check Flash Attention
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"

# Check transformers version
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Monitor GPU during inference
watch -n 0.5 nvidia-smi
```

### Known Issues

1. **Flash Attention wheel not available for Blackwell**
   - Status: As of Jan 2025, official wheels support up to sm_90 (Ada)
   - Workaround: Build from source with sm_100 support

2. **torch.compile with Flash Attention**
   - May cause errors on some GPUs
   - Disable if needed: Remove `use_torch_compile` flag

3. **Multi-GPU support**
   - Currently only uses GPU 0
   - For multi-GPU, modifications needed in `indextts/infer_v2_modded.py`

### Getting Help

If you still have issues:
1. Run with `--verbose` flag
2. Check the full error trace
3. Verify all diagnostics shown at startup
4. Share GPU info, error message, and startup diagnostics

### Changelog

**Latest improvements:**
- ✅ Added GPU compatibility checks at startup
- ✅ Added Flash Attention validation
- ✅ Added OOM error handling with automatic cache clearing
- ✅ Added memory usage information
- ✅ Added Blackwell-specific optimizations
- ✅ Added worker count suggestions
- ✅ Improved error messages with actionable advice
