# IndexTTS Performance Optimization Guide

Comprehensive guide to optimizing IndexTTS for maximum performance across different GPU architectures and configurations.

## Table of Contents

1. [Overview](#overview)
2. [Architecture-Specific Optimizations](#architecture-specific-optimizations)
3. [Memory Management](#memory-management)
4. [CUDA Graphs](#cuda-graphs)
5. [Mixed Precision Training](#mixed-precision-training)
6. [Performance Monitoring](#performance-monitoring)
7. [Batching Strategies](#batching-strategies)
8. [Advanced Optimizations](#advanced-optimizations)

## Overview

IndexTTS includes several layers of performance optimization:

- ✅ **Automatic architecture detection** and configuration
- ✅ **Memory cache management** with automatic eviction
- ✅ **Per-component timing telemetry**
- ✅ **CUDA graphs** for reduced kernel launch overhead
- ✅ **Mixed precision** (FP16/BF16/TF32) support
- ✅ **Flash Attention** integration for sm_80+ GPUs
- ✅ **Multi-GPU worker distribution**

### Performance Gains

Expected speedups with optimizations enabled:

| Optimization | Speedup | Requirements |
|-------------|---------|--------------|
| FP16/BF16 | 1.5-2.5x | Ampere+ GPU (sm_80+) |
| Flash Attention | 1.3-1.8x | Ampere+ with flash-attn installed |
| TF32 | 1.2-1.4x | Ampere+ GPU |
| CUDA Graphs | 1.1-1.3x | CUDA device, stable shapes |
| Cache Management | Variable | Prevents OOM, maintains throughput |

## Architecture-Specific Optimizations

### Automatic Configuration

The system automatically detects and configures optimal settings:

```python
from indextts import IndexTTS2

# Automatic optimization (recommended)
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True  # Enables optimal precision for your GPU
)
```

### Manual Configuration

```python
from indextts.utils.gpu_config import GPUConfig

# Check and apply optimal settings
settings = GPUConfig.apply_optimal_settings(device_id=0)
print(f"Applied settings: {settings}")

# Returns:
# {
#     'use_bf16': True/False,
#     'use_fp16': True/False,
#     'use_tf32': True/False,
#     'use_flash_attn': True/False,
#     'cudnn_benchmark': True,
#     'architecture': 'Ampere/Ada/Hopper/...'
# }
```

### Per-Architecture Settings

#### Blackwell (sm_100+) - RTX 6000 Ada, etc.

```python
# Automatic settings:
# - BFloat16 (BF16)
# - TensorFloat-32 (TF32) for matmul
# - Flash Attention v2
# - cuDNN benchmarking enabled

tts = IndexTTS2(use_fp16=True, device="cuda:0")
```

**Notes**:
- Flash Attention may require building from source
- Highest performance tier available

#### Hopper (sm_90) - H100, H800

```python
# Automatic settings:
# - BFloat16 (BF16)
# - TF32 enabled
# - Flash Attention v2
# - Transformer Engine support (if available)

tts = IndexTTS2(use_fp16=True, device="cuda:0")
```

**Notes**:
- Excellent BF16 performance
- FP8 support available but not currently utilized

#### Ada Lovelace (sm_89) - RTX 4090, RTX 6000 Ada

```python
# Automatic settings:
# - BFloat16 (BF16) preferred
# - FP16 fallback
# - TF32 enabled
# - Flash Attention v2

tts = IndexTTS2(use_fp16=True, device="cuda:0")
```

**Notes**:
- Excellent mixed-precision performance
- Great for consumer workstations

#### Ampere (sm_80-86) - A100, RTX 3090, A6000

```python
# Automatic settings:
# - BFloat16 (BF16) or FP16
# - TF32 enabled
# - Flash Attention v2

tts = IndexTTS2(use_fp16=True, device="cuda:0")
```

**Notes**:
- First architecture with BF16 support
- Widely available and well-optimized

#### Turing (sm_75) - RTX 2080 Ti, T4, Quadro RTX

```python
# Automatic settings:
# - FP16 only (no BF16)
# - Tensor Cores enabled
# - No Flash Attention

tts = IndexTTS2(use_fp16=True, device="cuda:0")
```

**Notes**:
- Solid FP16 performance
- No TF32 or Flash Attention support

#### Volta (sm_70) - V100

```python
# Automatic settings:
# - FP16 only
# - Basic Tensor Cores
# - No TF32 or Flash Attention

tts = IndexTTS2(use_fp16=True, device="cuda:0")
```

**Notes**:
- Oldest architecture with Tensor Cores
- Limited to FP16 optimizations

### Flash Attention Installation

For Ampere and newer GPUs (sm_80+):

```bash
# Standard installation (Ada and below)
uv sync --extra flashattn

# For Blackwell GPUs, build from source
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
MAX_JOBS=4 python setup.py install
```

Verify installation:

```python
from indextts.utils.gpu_config import GPUConfig
print(f"Flash Attention available: {GPUConfig.check_flash_attention()}")
```

## Memory Management

### Cache Configuration

IndexTTS includes intelligent cache management:

```python
# Configuration constants (in indextts/infer_v2_modded.py)
MAX_CACHE_SIZE_MB = 2048      # Maximum cache size in MB
CACHE_CHECK_INTERVAL = 5      # Check every N inferences
```

### Automatic Eviction

Cache is automatically checked and evicted when needed:

```python
# Happens automatically every CACHE_CHECK_INTERVAL inferences
tts.infer(text="Hello world", ...)  # Cache managed automatically
```

### Manual Cache Management

```python
# Check cache size
cache_mb = tts._get_cache_memory_mb()
print(f"Cache size: {cache_mb:.1f} MB")

# Force cache eviction if needed
if cache_mb > 1500:  # Custom threshold
    tts._check_and_evict_cache()

# Clear all caches completely
tts._clear_all_caches()
```

### Cache Types

The system caches several components:

1. **Speaker embeddings** (`cache_spk_cond`)
2. **Emotion embeddings** (`cache_emo_cond`)
3. **S2Mel style/prompts** (`cache_s2mel_style`, `cache_s2mel_prompt`)
4. **Mel spectrograms** (`cache_mel`)

### VRAM Monitoring

```python
from indextts.utils.resource_monitor import get_vram_summary, predict_oom_risk

# Check current VRAM usage
used_mb, total_mb, percent = get_vram_summary(device_id=0)
print(f"VRAM: {used_mb}/{total_mb} MB ({percent:.1f}%)")

# Predict OOM risk
risk = predict_oom_risk(device_id=0, threshold_percent=90)
if risk:
    print("WARNING: High OOM risk - consider clearing cache")
    tts._clear_all_caches()
```

## CUDA Graphs

CUDA graphs reduce kernel launch overhead by capturing and replaying operation sequences.

### Enabling CUDA Graphs

```python
from indextts import IndexTTS2

# Initialize TTS
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")

# Enable CUDA graphs (optional, advanced users only)
if tts.cuda_graphs_available:
    success = tts.enable_cuda_graphs(warmup_steps=3)
    if success:
        print("CUDA graphs enabled successfully")
    else:
        print("CUDA graphs failed to enable")
```

### How It Works

CUDA graphs are applied selectively to the vocoder (BigVGAN):

1. **Warmup**: System runs several warmup iterations with typical input shapes
2. **Capture**: On first inference, graph is captured for the input shape
3. **Replay**: Subsequent inferences with matching shapes use the cached graph

### Performance Characteristics

**Best for**:
- Repeated inferences with similar-length outputs
- Long-running services with stable workloads
- GPU-bound scenarios (not CPU-bound)

**Not recommended for**:
- Highly variable input lengths
- One-off generations
- Debugging/development

### Disabling CUDA Graphs

```python
# Disable if experiencing issues
tts.disable_cuda_graphs()
```

### Limitations

- Requires CUDA device (not available on CPU/MPS)
- Works best with stable input shapes
- May not capture benefits with highly variable sentence lengths
- Current implementation focuses on vocoder only (due to CFM solver constraints)

## Mixed Precision Training

### Precision Hierarchy

**BFloat16 (BF16)** - Best for most modern GPUs
- Available: Ampere+ (sm_80+)
- Benefits: Better numerical stability than FP16, wide dynamic range
- Use case: Training and inference on Ampere/Ada/Hopper/Blackwell

**Float16 (FP16)** - Legacy mixed precision
- Available: Volta+ (sm_70+)
- Benefits: 2x throughput vs FP32, widely supported
- Use case: Older GPUs (Volta, Turing) or fallback

**TensorFloat-32 (TF32)** - Automatic acceleration
- Available: Ampere+ (sm_80+)
- Benefits: Transparent acceleration for FP32 ops, no code changes
- Use case: Automatically enabled on compatible GPUs

**Float32 (FP32)** - Default precision
- Available: All GPUs
- Benefits: Maximum numerical stability
- Use case: CPU inference, debugging, reference results

### Enabling Mixed Precision

```python
# Automatic (recommended) - uses optimal precision for your GPU
tts = IndexTTS2(use_fp16=True)

# Check active precision
print(f"Active dtype: {tts.dtype}")  # torch.float16, torch.bfloat16, or None (fp32)
```

### Manual Precision Selection

```python
import torch
from indextts import IndexTTS2

# Force FP32 (debugging, maximum accuracy)
tts = IndexTTS2(use_fp16=False)

# Check if BF16 is available
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    bf16_available = (props.major, props.minor) >= (8, 0)
    print(f"BF16 available: {bf16_available}")
```

### Automatic Mixed Precision (AMP)

IndexTTS uses PyTorch AMP internally:

```python
# In inference code (automatic):
with torch.amp.autocast(device.type, enabled=self.dtype is not None, dtype=self.dtype):
    output = model(input)
```

## Performance Monitoring

### Telemetry System

Every inference collects detailed timing data:

```python
# Run inference
output_path = tts.infer(
    spk_audio_prompt="reference.wav",
    text="Hello, this is a test",
    output_path="output.wav"
)

# Access telemetry data
print(f"GPT generation time: {tts.last_gpt_time:.3f}s")
print(f"S2Mel generation time: {tts.last_s2mel_time:.3f}s")
print(f"BigVGAN vocoder time: {tts.last_bigvgan_time:.3f}s")
print(f"Total inference time: {tts.last_total_time:.3f}s")
print(f"Generated audio duration: {tts.last_audio_duration:.3f}s")
print(f"Real-Time Factor (RTF): {tts.last_rtf:.3f}x")
```

### Interpreting RTF

**Real-Time Factor (RTF)** = Generation Time / Audio Duration

- RTF < 1.0: Faster than real-time (good)
- RTF = 1.0: Generates as fast as playback
- RTF > 1.0: Slower than real-time

**Example**:
```
Audio duration: 5.0 seconds
Generation time: 2.5 seconds
RTF = 2.5 / 5.0 = 0.5x (2x faster than real-time)
```

### Component Breakdown

Typical time distribution:

```
GPT (Text→Codes):     40-50%
S2Mel (Codes→Mel):    30-40%
BigVGAN (Mel→Audio):  15-25%
Other (loading, I/O): 5-10%
```

### Bottleneck Identification

```python
# Run inference and analyze
tts.infer(text="Test", ...)

total = tts.last_total_time
gpt_pct = (tts.last_gpt_time / total) * 100
s2mel_pct = (tts.last_s2mel_time / total) * 100
bigvgan_pct = (tts.last_bigvgan_time / total) * 100

print(f"\nTime breakdown:")
print(f"  GPT:     {gpt_pct:.1f}%")
print(f"  S2Mel:   {s2mel_pct:.1f}%")
print(f"  BigVGAN: {bigvgan_pct:.1f}%")

if gpt_pct > 60:
    print("⚠️ GPT is bottleneck - consider using Flash Attention")
elif bigvgan_pct > 40:
    print("⚠️ Vocoder is bottleneck - consider enabling CUDA graphs")
```

### WebUI Monitoring

The WebUI includes real-time monitoring:

1. **GPU stats**: VRAM usage, temperature, utilization
2. **Generation history**: Track all generated samples
3. **Model metadata**: Size, vocab, creation date
4. **Performance stats**: RTF, generation times

## Batching Strategies

### Current Implementation

Due to model architecture constraints (CFM solver requires batch_size=1), dynamic batching is not currently used in the inference pipeline.

### Future Batching Support

A batching helper function is provided for future use:

```python
from indextts.infer_v2_modded import group_sentences_by_length

# Group sentences by similar length
sentences = [["hello", "world"], ["a"], ["test", "sentence", "here"]]
batches = group_sentences_by_length(
    sentences,
    tolerance=0.2,      # 20% length difference
    max_batch_size=4    # Max 4 sentences per batch
)
print(batches)  # [[1], [0], [2]] - grouped by length
```

### Workarounds for Throughput

Instead of batching, use parallelization:

```bash
# Run parallel WebUI for multi-GPU throughput
python webui_parallel.py

# Multiple workers process requests concurrently
```

## Advanced Optimizations

### Torch Compile (Experimental)

Enable PyTorch 2.x compilation for S2Mel:

```python
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_torch_compile=True  # Experimental
)
```

**Notes**:
- Requires PyTorch 2.0+
- Long first-run compilation time
- May provide 10-20% speedup after warmup
- Disable if experiencing issues

### DeepSpeed Integration

For training or large-scale inference:

```bash
# Enable DeepSpeed
export INDEXTTS_USE_DEEPSPEED=1

# Disable DeepSpeed
export INDEXTTS_USE_DEEPSPEED=0
```

```python
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_deepspeed=True  # Explicit control
)
```

### cuDNN Autotuner

Automatically enabled for optimal convolution algorithms:

```python
# Happens automatically in GPUConfig.apply_optimal_settings()
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # Enable autotuner
```

**Notes**:
- First few runs will be slower (algorithm search)
- Subsequent runs benefit from optimal algorithm selection
- Disable for debugging: `cudnn.benchmark = False`

### Memory-Efficient Attention

For GPUs without Flash Attention:

```python
# PyTorch's memory-efficient attention (automatic fallback)
# Used when Flash Attention is not available
```

## Performance Checklist

Use this checklist to ensure optimal performance:

### Pre-Flight Check

- [ ] GPU architecture detected correctly (`GPUConfig.list_available_gpus()`)
- [ ] Mixed precision enabled (`use_fp16=True` on capable GPUs)
- [ ] Flash Attention installed for sm_80+ GPUs
- [ ] Sufficient VRAM available (8-12GB recommended)
- [ ] CUDA version compatible with PyTorch (check `nvidia-smi`)

### During Inference

- [ ] Monitor VRAM usage (`get_vram_summary()`)
- [ ] Check RTF < 1.0 for real-time capability
- [ ] Review telemetry for bottlenecks
- [ ] Cache eviction working (no OOM errors)
- [ ] Temperature staying within limits (<85°C)

### Optimization Opportunities

- [ ] Enable CUDA graphs for repeated patterns
- [ ] Use parallel WebUI for multi-request throughput
- [ ] Batch similar-length requests together (manual parallelization)
- [ ] Profile with NVIDIA Nsight for deep analysis
- [ ] Consider torch.compile for S2Mel (experimental)

## Profiling Tools

### PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    tts.infer(text="Test", ...)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### NVIDIA Nsight

```bash
# Profile with Nsight Systems
nsys profile -o output python your_script.py

# Profile with Nsight Compute
ncu --set full -o output python your_script.py
```

### Memory Profiling

```python
import torch

# Start memory profiling
torch.cuda.reset_peak_memory_stats()

# Run inference
tts.infer(text="Test", ...)

# Check peak memory
peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak VRAM: {peak_mb:.1f} MB")
```

## Benchmarking

### Standard Benchmark

```python
import time
import numpy as np

# Warmup
for _ in range(3):
    tts.infer(text="Warmup run", ...)

# Benchmark
times = []
for i in range(10):
    start = time.perf_counter()
    tts.infer(text=f"Benchmark run {i}", ...)
    times.append(time.perf_counter() - start)

print(f"Mean: {np.mean(times):.3f}s")
print(f"Std: {np.std(times):.3f}s")
print(f"Min: {np.min(times):.3f}s")
print(f"Max: {np.max(times):.3f}s")
```

### Comparative Benchmark

```python
# Test with/without optimizations
configs = [
    {"use_fp16": False, "name": "FP32 baseline"},
    {"use_fp16": True, "name": "FP16 optimized"},
]

for config in configs:
    tts = IndexTTS2(**config)
    # ... run benchmark ...
    print(f"{config['name']}: {mean_time:.3f}s (RTF: {rtf:.2f}x)")
```

## Troubleshooting Performance

### Slow Inference

**Check**:
1. Mixed precision enabled?
2. Flash Attention installed (for sm_80+)?
3. GPU utilization low? (may indicate CPU bottleneck)
4. Running on CPU by mistake?

**Solutions**:
```python
# Verify device
print(f"Device: {tts.device}")  # Should be cuda:X

# Check GPU utilization
from indextts.utils.resource_monitor import ResourceMonitor
monitor = ResourceMonitor()
stats = monitor.get_gpu_stats(0)
print(f"GPU utilization: {stats.utilization_percent}%")  # Should be >70%
```

### Variable Performance

**Symptoms**: Inconsistent inference times

**Causes**:
- Cache misses (different reference audio each time)
- Variable text lengths
- Thermal throttling
- Background GPU usage

**Solutions**:
- Use consistent reference audio for caching
- Monitor temperature (`stats.temperature_c`)
- Check for other GPU processes (`nvidia-smi`)

### High Memory Usage

**Solutions**:
1. Reduce cache size: `MAX_CACHE_SIZE_MB = 1024`
2. Check more frequently: `CACHE_CHECK_INTERVAL = 2`
3. Disable cache warmup: `tts._clear_all_caches()`

## Additional Resources

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Mixed Precision Training Guide](https://pytorch.org/docs/stable/amp.html)
- [IndexTTS Multi-GPU Guide](./MULTI_GPU_GUIDE.md)
