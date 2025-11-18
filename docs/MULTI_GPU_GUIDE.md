# Multi-GPU Setup and Usage Guide

This guide explains how to configure and use IndexTTS with multiple GPUs for improved throughput and performance.

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Architecture Detection](#architecture-detection)
4. [Multi-GPU Configuration](#multi-gpu-configuration)
5. [WebUI Parallel Mode](#webui-parallel-mode)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

## Overview

IndexTTS supports multi-GPU processing through two mechanisms:

- **Round-robin worker assignment**: Distributes inference workers across available GPUs
- **Architecture-specific optimizations**: Automatically configures optimal settings per GPU generation

### Supported GPU Architectures

| Architecture | Compute Capability | Features | Example GPUs |
|-------------|-------------------|----------|--------------|
| Blackwell | sm_100+ | BF16, TF32, Flash Attention | RTX 6000 Ada |
| Hopper | sm_90 | BF16, TF32, Flash Attention | H100, H800 |
| Ada Lovelace | sm_89 | BF16, FP16, TF32, Flash Attention | RTX 4090, RTX 6000 Ada |
| Ampere | sm_80-86 | BF16, FP16, TF32, Flash Attention | A100, RTX 3090, A6000 |
| Turing | sm_75 | FP16, Tensor Cores | RTX 2080 Ti, T4 |
| Volta | sm_70 | FP16 | V100 |

## System Requirements

### Hardware

- **VRAM**: 8-12GB per GPU recommended
- **Multiple GPUs**: At least 2 CUDA-capable GPUs
- **Mixed architectures**: Supported (e.g., RTX 3090 + RTX 4090)

### Software

```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## Architecture Detection

IndexTTS automatically detects your GPU architecture and applies optimal settings.

### Check Your GPU Configuration

```python
from indextts.utils.gpu_config import GPUConfig

# List all available GPUs
GPUConfig.list_available_gpus()

# Get optimal settings for GPU 0
settings = GPUConfig.apply_optimal_settings(device_id=0)
print(f"GPU 0 settings: {settings}")
```

### Manual Architecture Check

```python
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Compute: {props.major}.{props.minor}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
```

## Multi-GPU Configuration

### Basic Setup

The system automatically distributes workers across available GPUs:

```python
from indextts import IndexTTS2

# Initialize with automatic GPU detection
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True  # Recommended for GPUs with sm_80+
)

# Check which GPU is being used
print(f"Using device: {tts.device}")
```

### Explicit GPU Selection

```python
# Use specific GPU
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    device="cuda:1"  # Use GPU 1
)
```

### Environment Variables

```bash
# Limit visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2

# Run with specific GPU
CUDA_VISIBLE_DEVICES=2 python webui.py
```

## WebUI Parallel Mode

The parallel WebUI (`webui_parallel.py`) automatically distributes work across multiple GPUs.

### Starting Parallel Mode

```bash
# Start parallel WebUI (automatically uses all available GPUs)
python webui_parallel.py

# Access at http://localhost:7862
```

### Worker Assignment Strategy

Workers are assigned to GPUs using round-robin distribution:

```
Worker 0 → GPU 0
Worker 1 → GPU 1
Worker 2 → GPU 2
Worker 3 → GPU 0  (wraps around)
Worker 4 → GPU 1
...
```

### Configuration

In `webui_parallel.py`, worker count is automatically set based on available GPUs:

```python
# Automatic configuration (recommended)
num_workers = min(torch.cuda.device_count() * 2, 8)

# Manual configuration
manager.start_locked(count=4)  # 4 workers
```

### GPU Selection in WebUI

The WebUI includes GPU monitoring and selection:

1. **GPU Configuration & Monitoring** accordion
2. Select GPU from dropdown
3. Click "Switch GPU" to change active GPU
4. Monitor VRAM, temperature, and utilization in real-time

## Performance Tuning

### VRAM Management

```python
from indextts.utils.resource_monitor import get_vram_summary

# Check VRAM usage
used_mb, total_mb, percent = get_vram_summary(device_id=0)
print(f"GPU 0 VRAM: {used_mb}/{total_mb} MB ({percent:.1f}%)")
```

### Cache Configuration

The inference engine includes automatic cache management:

```python
# Cache settings (in infer_v2_modded.py)
MAX_CACHE_SIZE_MB = 2048  # Maximum cache size
CACHE_CHECK_INTERVAL = 5   # Check every N inferences

# Clear cache manually if needed
tts._clear_all_caches()
```

### Mixed Precision

Different architectures support different precision modes:

```python
# Automatic (recommended)
tts = IndexTTS2(use_fp16=True)  # Uses optimal precision for your GPU

# Check active precision
print(f"Using dtype: {tts.dtype}")  # torch.float16 or None (fp32)
```

### Optimization by Architecture

**Blackwell/Hopper/Ada (sm_89+)**:
- Use BF16 for best performance
- Enable Flash Attention
- TF32 enabled automatically

**Ampere (sm_80-86)**:
- Use BF16 or FP16
- Flash Attention available
- TF32 enabled for matmul

**Turing (sm_75)**:
- Use FP16
- Tensor Cores enabled
- No Flash Attention

**Volta (sm_70)**:
- FP16 only
- Basic Tensor Cores

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory errors

**Solutions**:

1. **Reduce batch size** (already set to 1 in current architecture)

2. **Clear cache more frequently**:
   ```python
   CACHE_CHECK_INTERVAL = 2  # Check more often
   MAX_CACHE_SIZE_MB = 1024  # Reduce cache size
   ```

3. **Use smaller segments**:
   ```python
   tts.infer(
       text=text,
       max_text_tokens_per_sentence=80  # Default is 120
   )
   ```

4. **Disable cache** (slower but less memory):
   ```python
   tts._clear_all_caches()
   ```

### Mixed Architecture Issues

**Symptoms**: Performance inconsistency across GPUs

**Solutions**:

1. **Check compute capabilities**:
   ```python
   from indextts.utils.gpu_config import GPUConfig
   GPUConfig.list_available_gpus()
   ```

2. **Use consistent precision** across GPUs:
   ```python
   # Force FP32 for compatibility
   tts = IndexTTS2(use_fp16=False)
   ```

3. **Balance workers** based on GPU capabilities:
   ```python
   # Assign more workers to faster GPUs manually
   # GPU 0: RTX 4090 (faster) → 3 workers
   # GPU 1: RTX 3090 (slower) → 2 workers
   ```

### Flash Attention Issues

**Symptoms**: Import errors or crashes with flash-attn

**Solutions**:

1. **Install flash-attn** (optional but recommended for sm_80+):
   ```bash
   # For most GPUs (up to Ada/sm_90)
   uv sync --extra flashattn

   # For Blackwell GPUs, build from source
   git clone https://github.com/Dao-AILab/flash-attention
   cd flash-attention
   MAX_JOBS=4 python setup.py install
   ```

2. **Check if Flash Attention is enabled**:
   ```python
   from indextts.utils.gpu_config import GPUConfig
   available = GPUConfig.check_flash_attention()
   print(f"Flash Attention available: {available}")
   ```

### Worker Crashes

**Symptoms**: Workers hang or crash in parallel mode

**Solutions**:

1. **Reduce worker count**:
   ```python
   num_workers = torch.cuda.device_count()  # One per GPU
   ```

2. **Check VRAM per GPU**:
   ```python
   from indextts.utils.resource_monitor import get_vram_summary
   for i in range(torch.cuda.device_count()):
       used, total, pct = get_vram_summary(i)
       print(f"GPU {i}: {pct:.1f}% used")
   ```

3. **Increase timeout** in `webui_parallel.py`:
   ```python
   # Increase worker timeout for slower GPUs
   timeout = 120  # seconds
   ```

### Performance Monitoring

Monitor your multi-GPU setup:

```python
from indextts.utils.resource_monitor import ResourceMonitor

monitor = ResourceMonitor()

# Get stats for all GPUs
for i in range(torch.cuda.device_count()):
    stats = monitor.get_gpu_stats(i)
    if stats:
        print(f"\nGPU {i}: {stats.name}")
        print(f"  VRAM: {stats.used_memory_mb:.0f}/{stats.total_memory_mb:.0f} MB")
        print(f"  Temp: {stats.temperature_c}°C")
        print(f"  Util: {stats.utilization_percent}%")
```

### Telemetry Data

Access per-component timing after inference:

```python
# Run inference
tts.infer(text="Hello world", ...)

# Check timing breakdown
print(f"GPT time: {tts.last_gpt_time:.3f}s")
print(f"S2Mel time: {tts.last_s2mel_time:.3f}s")
print(f"BigVGAN time: {tts.last_bigvgan_time:.3f}s")
print(f"Total time: {tts.last_total_time:.3f}s")
print(f"RTF (Real-Time Factor): {tts.last_rtf:.3f}x")
```

## Best Practices

1. **Use matched GPU architectures** when possible for consistent performance
2. **Enable FP16/BF16** on Ampere+ GPUs for 2x performance
3. **Monitor VRAM** usage to prevent OOM crashes
4. **Balance workers** based on actual GPU capabilities
5. **Use Flash Attention** on supported GPUs (sm_80+) for transformer speedup
6. **Check telemetry** regularly to identify bottlenecks

## Additional Resources

- [PyTorch Multi-GPU Best Practices](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
- [Flash Attention Repository](https://github.com/Dao-AILab/flash-attention)
- [IndexTTS Optimization Guide](./OPTIMIZATION_GUIDE.md)
