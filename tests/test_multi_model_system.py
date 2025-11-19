#!/usr/bin/env python3
"""
Test script for multi-model system components.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from indextts.utils.model_manager import ModelManager, ModelMetadata
        print("✓ ModelManager imported successfully")
    except Exception as e:
        print(f"✗ ModelManager import failed: {e}")
        return False

    try:
        from indextts.utils.model_comparison import ModelComparator, GenerationMetrics, ComparisonResult
        print("✓ ModelComparator imported successfully")
    except Exception as e:
        print(f"✗ ModelComparator import failed: {e}")
        return False

    return True


def test_model_manager():
    """Test ModelManager basic functionality."""
    print("\nTesting ModelManager...")

    from indextts.utils.model_manager import ModelManager

    try:
        # Initialize ModelManager
        manager = ModelManager()
        print("✓ ModelManager initialized")

        # Check GPU memory info
        mem_info = manager.get_memory_usage()
        print(f"✓ GPU Memory: Allocated={mem_info['allocated_gb']:.2f}GB, "
              f"Reserved={mem_info['reserved_gb']:.2f}GB, "
              f"Free={mem_info['free_gb']:.2f}GB")

        # List models in registry
        models = manager.list_models()
        print(f"✓ Found {len(models)} models in registry")

        return True
    except Exception as e:
        print(f"✗ ModelManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_metadata_extraction():
    """Test metadata extraction for available models."""
    print("\nTesting model metadata extraction...")

    from indextts.utils.model_manager import ModelManager
    from pathlib import Path

    try:
        manager = ModelManager()

        # Find a GPT checkpoint to test
        checkpoints_dir = Path("checkpoints")
        if not checkpoints_dir.exists():
            print("⚠ No checkpoints directory found, skipping metadata test")
            return True

        gpt_files = list(checkpoints_dir.glob("*.pth"))
        if not gpt_files:
            print("⚠ No .pth files found, skipping metadata test")
            return True

        test_model = str(gpt_files[0])
        print(f"Testing metadata extraction for: {Path(test_model).name}")

        metadata = manager.extract_model_metadata(test_model)
        print(f"✓ Model: {metadata.filename}")
        print(f"  Size: {metadata.size_mb:.1f} MB")
        print(f"  Version: {metadata.version}")
        print(f"  Languages: {', '.join(metadata.languages)}")
        print(f"  Vocab Size: {metadata.vocab_size}")
        print(f"  Model Dim: {metadata.model_dim}")
        print(f"  Layers: {metadata.num_layers}")
        print(f"  Heads: {metadata.num_heads}")
        print(f"  Architecture: {metadata.architecture}")
        print(f"  Tokenizer: {metadata.tokenizer_path}")
        print(f"  Estimated VRAM: {metadata.recommended_vram_gb:.1f} GB")

        return True
    except Exception as e:
        print(f"✗ Metadata extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_comparator():
    """Test ModelComparator initialization."""
    print("\nTesting ModelComparator...")

    from indextts.utils.model_manager import ModelManager
    from indextts.utils.model_comparison import ModelComparator

    try:
        manager = ModelManager()
        comparator = ModelComparator(manager)
        print("✓ ModelComparator initialized successfully")

        return True
    except Exception as e:
        print(f"✗ ModelComparator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_persistence():
    """Test model persistence features (mark_activity, get_status, etc)."""
    print("\nTesting model persistence features...")

    from indextts.utils.model_manager import ModelManager
    import time

    try:
        # Test with auto-unload disabled (default)
        manager = ModelManager()

        # Test get_status with no model loaded
        status = manager.get_status()
        assert status['model_loaded'] == False, "Model should not be loaded initially"
        assert status['idle_time_seconds'] is None, "Idle time should be None when no model loaded"
        print("✓ get_status() works with no model loaded")

        # Test get_idle_time
        idle_time = manager.get_idle_time()
        assert idle_time is None, "Idle time should be None when no model loaded"
        print("✓ get_idle_time() returns None when no model loaded")

        # Test mark_activity (should be no-op when no model loaded)
        manager.mark_activity()
        print("✓ mark_activity() works when no model loaded")

        # Test VRAM warning
        warning = manager.get_vram_warning()
        # Warning could be None or a string depending on VRAM usage
        print(f"✓ get_vram_warning() works: {warning if warning else 'No warning'}")

        # Test auto-unload manager
        manager_auto = ModelManager(auto_unload_timeout=10)
        auto_status = manager_auto.get_status()
        assert auto_status['auto_unload_enabled'] == True, "Auto-unload should be enabled"
        assert auto_status['auto_unload_timeout'] == 10, "Auto-unload timeout should be 10s"
        print("✓ Auto-unload configuration works")

        # Test check_auto_unload (should return False when no model loaded)
        unloaded = manager_auto.check_auto_unload()
        assert unloaded == False, "Should not unload when no model is loaded"
        print("✓ check_auto_unload() returns False when no model loaded")

        return True
    except Exception as e:
        print(f"✗ Model persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Model System Test Suite")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("ModelManager Basic", test_model_manager),
        ("Metadata Extraction", test_model_metadata_extraction),
        ("ModelComparator Init", test_model_comparator),
        ("Model Persistence", test_model_persistence),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)
