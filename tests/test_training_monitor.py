#!/usr/bin/env python3
"""
Test suite for Training Monitor utilities.
"""
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.utils.training_monitor import (
    TrainingMetrics,
    TrainingLogParser,
    TrainingAnalyzer,
    TensorBoardManager,
    find_training_logs,
    get_tensorboard_logdir
)


def test_training_metrics():
    """Test TrainingMetrics dataclass."""
    print("=" * 70)
    print("TEST: TrainingMetrics")
    print("=" * 70)

    metrics = TrainingMetrics(
        step=100,
        epoch=1,
        loss=2.345,
        learning_rate=1.5e-4,
        grad_norm=1.234,
        time_per_step=0.5,
        vram_gb=12.3
    )

    assert metrics.step == 100
    assert metrics.epoch == 1
    assert metrics.loss == 2.345
    assert metrics.learning_rate == 1.5e-4

    print(f"✅ TrainingMetrics created successfully")
    print(f"  Step: {metrics.step}")
    print(f"  Loss: {metrics.loss}")
    print(f"  LR: {metrics.learning_rate}")
    return True


def test_log_parser():
    """Test TrainingLogParser with mock log data."""
    print("\n" + "=" * 70)
    print("TEST: TrainingLogParser")
    print("=" * 70)

    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_path = Path(f.name)

        # Write mock log data
        f.write("Step 100/5000 | Epoch 1/10 | Loss: 2.345 | LR: 1.5e-4\n")
        f.write("Step 200/5000 | Epoch 1/10 | Loss: 2.123 | LR: 1.5e-4\n")
        f.write("Step 300/5000 | Epoch 2/10 | Loss: 1.987 | LR: 1.4e-4\n")

    try:
        parser = TrainingLogParser(log_path)
        metrics = parser.parse_log_file()

        assert len(metrics) == 3, f"Expected 3 metrics, got {len(metrics)}"
        assert metrics[0].step == 100
        assert metrics[0].loss == 2.345
        assert metrics[1].step == 200
        assert metrics[2].epoch == 2

        print(f"✅ Parsed {len(metrics)} metrics successfully")
        for m in metrics:
            print(f"  Step {m.step}: Loss={m.loss:.3f}, LR={m.learning_rate:.2e}")

        # Test recent lines parsing
        recent = parser.parse_recent_lines(2)
        assert len(recent) >= 1, f"Expected at least 1 recent metric, got {len(recent)}"
        print(f"✅ Recent lines parsing works (got {len(recent)} recent metrics)")

        # Test get_latest
        latest = parser.get_latest_metrics()
        assert latest.step == 300
        print(f"✅ Latest metrics: Step {latest.step}, Loss {latest.loss:.3f}")

        return True
    finally:
        log_path.unlink()


def test_training_analyzer():
    """Test TrainingAnalyzer detection algorithms."""
    print("\n" + "=" * 70)
    print("TEST: TrainingAnalyzer")
    print("=" * 70)

    analyzer = TrainingAnalyzer()

    # Create mock metrics with plateau
    plateau_metrics = [
        TrainingMetrics(i, 1, 2.0 + (i % 3) * 0.001, 1e-4)
        for i in range(100)
    ]

    is_plateau, msg = analyzer.detect_plateau(plateau_metrics, patience=50)
    print(f"Plateau detection: {is_plateau}")
    print(f"  Message: {msg}")
    print(f"  ✅ Plateau detection working")

    # Create mock metrics with divergence
    divergence_metrics = [
        TrainingMetrics(i, 1, 2.0 + i * 0.5, 1e-4)
        for i in range(20)
    ]

    is_div, div_msg = analyzer.detect_divergence(divergence_metrics, threshold=10.0)
    print(f"\nDivergence detection: {is_div}")
    print(f"  Message: {div_msg}")
    print(f"  ✅ Divergence detection working")

    # Test time estimation with timing data
    timed_metrics = [
        TrainingMetrics(i, 1, 2.0, 1e-4, time_per_step=0.5)
        for i in range(100)
    ]

    seconds, time_msg = analyzer.estimate_time_remaining(timed_metrics, target_step=200)
    print(f"\nTime estimation:")
    print(f"  {time_msg}")
    print(f"  Seconds remaining: {seconds}")
    assert seconds is not None
    print(f"  ✅ Time estimation working")

    # Test run comparison
    runs = {
        'run1': [TrainingMetrics(i, 1, 2.0 - i * 0.01, 1e-4) for i in range(50)],
        'run2': [TrainingMetrics(i, 1, 2.5 - i * 0.015, 1e-4) for i in range(50)],
    }

    comparison = analyzer.compare_runs(runs)
    print(f"\nRun comparison:")
    print(f"  Run 1 best loss: {comparison['run1']['best_loss']:.4f}")
    print(f"  Run 2 best loss: {comparison['run2']['best_loss']:.4f}")
    print(f"  Winner: {comparison['_best_run']}")
    assert '_best_run' in comparison
    print(f"  ✅ Run comparison working")

    return True


def test_tensorboard_manager():
    """Test TensorBoardManager (without actually starting TB)."""
    print("\n" + "=" * 70)
    print("TEST: TensorBoardManager")
    print("=" * 70)

    manager = TensorBoardManager()

    # Test initialization
    assert manager.tensorboard_process is None
    assert manager.tensorboard_port is None
    print(f"✅ TensorBoardManager initialized")

    # Test is_running (should be False initially)
    assert not manager.is_running()
    print(f"✅ is_running() works (returns False initially)")

    # Test get_url (should be None initially)
    assert manager.get_url() is None
    print(f"✅ get_url() works (returns None initially)")

    return True


def test_helper_functions():
    """Test helper functions."""
    print("\n" + "=" * 70)
    print("TEST: Helper Functions")
    print("=" * 70)

    # Test get_tensorboard_logdir
    logdir = get_tensorboard_logdir("test_project")
    assert logdir == Path("training/test_project/checkpoints/runs")
    print(f"✅ get_tensorboard_logdir works: {logdir}")

    # Test find_training_logs with non-existent project
    logs = find_training_logs("nonexistent_project")
    assert logs == []
    print(f"✅ find_training_logs handles missing projects correctly")

    return True


def main():
    """Run all tests."""
    print("\n🧪 Testing Training Monitor\n")

    tests = [
        ("TrainingMetrics", test_training_metrics),
        ("LogParser", test_log_parser),
        ("TrainingAnalyzer", test_training_analyzer),
        ("TensorBoardManager", test_tensorboard_manager),
        ("Helper Functions", test_helper_functions),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTests passed: {passed}/{total}")

    if all(result for _, result in results):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
