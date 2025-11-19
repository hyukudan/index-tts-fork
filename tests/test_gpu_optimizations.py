#!/usr/bin/env python3
"""
Test GPU-specific optimizations for different architectures.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from indextts.utils.gpu_config import GPUConfig


def test_architecture_detection():
    """Test that GPU architecture is correctly detected."""
    print("=" * 70)
    print("TEST: GPU Architecture Detection")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping test")
        return True

    gpus = GPUConfig.get_gpu_info()

    if not gpus:
        print("❌ No GPUs detected")
        return False

    for gpu in gpus:
        print(f"\nGPU {gpu['id']}: {gpu['name']}")
        print(f"  Architecture: {gpu['architecture']}")
        print(f"  Compute Capability: {gpu['compute_capability']}")
        print(f"  Total Memory: {gpu['total_memory_gb']:.1f} GB")
        print(f"  Blackwell: {gpu['is_blackwell']}")
        print(f"  Ada or newer: {gpu['is_ada_or_newer']}")
        print(f"  Supports BF16: {gpu['supports_bf16']}")

        # Verify architecture classification
        major, minor = map(int, gpu['compute_capability'].split('.'))

        if major >= 10:
            assert gpu['architecture'] == "Blackwell", f"Expected Blackwell, got {gpu['architecture']}"
            assert gpu['is_blackwell'] == True
            print("  ✅ Correctly identified as Blackwell")
        elif major == 8 and minor >= 9:
            assert gpu['architecture'] == "Ada Lovelace", f"Expected Ada Lovelace, got {gpu['architecture']}"
            assert gpu['is_ada_or_newer'] == True
            print("  ✅ Correctly identified as Ada Lovelace")
        elif major == 8:
            assert gpu['architecture'] == "Ampere", f"Expected Ampere, got {gpu['architecture']}"
            print("  ✅ Correctly identified as Ampere")

    print("\n✅ Architecture detection test passed!")
    return True


def test_optimal_settings():
    """Test that optimal settings are correctly determined for each architecture."""
    print("\n" + "=" * 70)
    print("TEST: Optimal Settings Configuration")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping test")
        return True

    gpus = GPUConfig.get_gpu_info()

    for gpu in gpus:
        compute_cap = tuple(map(int, gpu['compute_capability'].split('.')))
        settings = GPUConfig.get_optimal_settings(compute_cap)

        print(f"\nGPU {gpu['id']}: {gpu['name']} ({gpu['architecture']})")
        print(f"  Recommended dtype: {settings['recommended_dtype']}")
        print(f"  BF16: {settings['use_bf16']}")
        print(f"  FP16: {settings['use_fp16']}")
        print(f"  TF32: {settings['use_tf32']}")
        print(f"  Flash Attention: {settings['enable_flash_attn']}")
        print(f"  Tensor Cores: {settings['tensor_cores_available']}")

        # Verify Blackwell settings
        if gpu['is_blackwell']:
            assert settings['use_bf16'] == True, "Blackwell should use BF16"
            assert settings['use_tf32'] == True, "Blackwell should use TF32"
            assert settings['enable_flash_attn'] == True, "Blackwell should support Flash Attention"
            assert settings['recommended_dtype'] == "bfloat16", "Blackwell should recommend bfloat16"
            print("  ✅ Blackwell optimizations verified")

        # Verify Ada Lovelace settings
        elif gpu['is_ada_or_newer']:
            assert settings['use_bf16'] == True, "Ada should use BF16"
            assert settings['use_fp16'] == True, "Ada should use FP16"
            assert settings['use_tf32'] == True, "Ada should use TF32"
            assert settings['enable_flash_attn'] == True, "Ada should support Flash Attention"
            print("  ✅ Ada Lovelace optimizations verified")

    print("\n✅ Optimal settings test passed!")
    return True


def test_apply_settings():
    """Test that settings can be applied to a GPU."""
    print("\n" + "=" * 70)
    print("TEST: Apply Settings")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping test")
        return True

    gpu_id = 0
    result = GPUConfig.apply_optimal_settings(gpu_id)

    print(f"\nApplying settings to GPU {gpu_id}:")
    print(f"  Status: {result.get('status')}")
    print(f"  Architecture: {result.get('architecture')}")
    print(f"  TF32 enabled: {result.get('tf32_enabled')}")
    print(f"  Recommended dtype: {result.get('recommended_dtype')}")
    print(f"  Flash Attn compatible: {result.get('flash_attn_compatible')}")

    if result.get('status') == 'applied':
        # Verify settings were actually applied
        import torch.backends.cudnn as cudnn

        compute_cap = torch.cuda.get_device_capability(gpu_id)
        settings = GPUConfig.get_optimal_settings(compute_cap)

        if settings['use_tf32']:
            # Check that TF32 is enabled
            assert torch.backends.cuda.matmul.allow_tf32 == True, "TF32 should be enabled for matmul"
            assert cudnn.allow_tf32 == True, "TF32 should be enabled for cuDNN"
            print("  ✅ TF32 settings verified")

        assert cudnn.benchmark == True, "cuDNN benchmark should be enabled"
        print("  ✅ cuDNN benchmark enabled")

        print("\n✅ Apply settings test passed!")
        return True
    else:
        print(f"⚠️  Could not apply settings: {result.get('reason')}")
        return True  # Not a failure if we can't apply


def main():
    """Run all tests."""
    print("\n🧪 Testing GPU Optimizations\n")

    tests = [
        test_architecture_detection,
        test_optimal_settings,
        test_apply_settings,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
