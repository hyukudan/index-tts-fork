#!/usr/bin/env python3
"""
Test suite for Emotion Presets utility.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.utils.emotion_presets import (
    EmotionPreset,
    EMOTION_PRESETS,
    get_preset_choices,
    get_preset_vector,
    mix_emotions,
    normalize_vector,
    get_all_presets,
    get_preset_description
)


def test_emotion_preset_dataclass():
    """Test EmotionPreset dataclass creation."""
    print("=" * 70)
    print("TEST: EmotionPreset Dataclass")
    print("=" * 70)

    preset = EmotionPreset(
        name="test",
        display_name="Test Emotion",
        emoji="😊",
        description="A test emotion",
        vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
    )

    assert preset.name == "test"
    assert preset.display_name == "Test Emotion"
    assert preset.emoji == "😊"
    assert len(preset.vector) == 8
    assert preset.vector[0] == 1.0

    print("✅ EmotionPreset dataclass works correctly")
    return True


def test_preset_definitions():
    """Test that all presets are properly defined."""
    print("\n" + "=" * 70)
    print("TEST: Preset Definitions")
    print("=" * 70)

    expected_presets = [
        "neutral", "happy", "excited", "sad", "angry",
        "fearful", "calm", "surprised", "disgusted",
        "confident", "tender"
    ]

    for preset_id in expected_presets:
        assert preset_id in EMOTION_PRESETS, f"Missing preset: {preset_id}"
        preset = EMOTION_PRESETS[preset_id]

        # Check all fields are present
        assert preset.name == preset_id
        assert len(preset.display_name) > 0
        assert len(preset.emoji) > 0
        assert len(preset.description) > 0
        assert len(preset.vector) == 8, f"Preset {preset_id} has wrong vector length"

        # Check vector values are in valid range [0, 1]
        for i, val in enumerate(preset.vector):
            assert 0.0 <= val <= 1.0, f"Preset {preset_id} vector[{i}]={val} out of range"

    print(f"✅ All {len(expected_presets)} presets are properly defined")
    print(f"   Presets: {', '.join(expected_presets)}")
    return True


def test_get_preset_choices():
    """Test get_preset_choices for Gradio dropdown."""
    print("\n" + "=" * 70)
    print("TEST: get_preset_choices()")
    print("=" * 70)

    choices = get_preset_choices()

    # Should have 11 presets + 1 custom option
    assert len(choices) == 12, f"Expected 12 choices, got {len(choices)}"

    # Check format: list of (label, value) tuples
    for choice in choices:
        assert isinstance(choice, tuple)
        assert len(choice) == 2
        label, value = choice
        assert isinstance(label, str)
        assert isinstance(value, str)

    # Check custom option is last
    assert choices[-1][1] == "custom"
    assert "Custom" in choices[-1][0]

    # Check emojis are in labels
    for label, _ in choices[:-1]:  # Exclude custom
        assert any(c for c in label if ord(c) > 127), f"No emoji in label: {label}"

    print(f"✅ get_preset_choices() returns {len(choices)} valid choices")
    return True


def test_get_preset_vector():
    """Test get_preset_vector retrieval."""
    print("\n" + "=" * 70)
    print("TEST: get_preset_vector()")
    print("=" * 70)

    # Test valid preset
    neutral_vec = get_preset_vector("neutral")
    assert neutral_vec is not None
    assert len(neutral_vec) == 8
    print(f"✅ neutral preset: {neutral_vec}")

    happy_vec = get_preset_vector("happy")
    assert happy_vec is not None
    assert happy_vec[0] > 0.5, "Happy should have high happiness"
    print(f"✅ happy preset: {happy_vec}")

    # Test invalid preset
    invalid_vec = get_preset_vector("nonexistent")
    assert invalid_vec is None
    print(f"✅ Invalid preset returns None")

    return True


def test_mix_emotions():
    """Test emotion mixing functionality."""
    print("\n" + "=" * 70)
    print("TEST: mix_emotions()")
    print("=" * 70)

    # Mix neutral (all zeros except calm=0.5) and happy
    neutral_vec = get_preset_vector("neutral")
    happy_vec = get_preset_vector("happy")

    # 50/50 mix
    mixed_50 = mix_emotions("neutral", "happy", 0.5)
    assert len(mixed_50) == 8

    # Check that mixed values are between neutral and happy
    for i in range(8):
        min_val = min(neutral_vec[i], happy_vec[i])
        max_val = max(neutral_vec[i], happy_vec[i])
        assert min_val <= mixed_50[i] <= max_val, \
            f"Mixed value [{i}]={mixed_50[i]} not in range [{min_val}, {max_val}]"

    print(f"✅ 50/50 mix: {mixed_50}")

    # 100% A (ratio=0.0)
    mixed_a = mix_emotions("neutral", "happy", 0.0)
    for i in range(8):
        assert abs(mixed_a[i] - neutral_vec[i]) < 0.001

    print(f"✅ 100% A mix matches first preset")

    # 100% B (ratio=1.0)
    mixed_b = mix_emotions("neutral", "happy", 1.0)
    for i in range(8):
        assert abs(mixed_b[i] - happy_vec[i]) < 0.001

    print(f"✅ 100% B mix matches second preset")

    # Test with invalid preset (should return neutral)
    mixed_invalid = mix_emotions("invalid", "happy", 0.5)
    assert mixed_invalid == EMOTION_PRESETS["neutral"].vector

    print(f"✅ Invalid preset falls back to neutral")

    return True


def test_normalize_vector():
    """Test vector normalization."""
    print("\n" + "=" * 70)
    print("TEST: normalize_vector()")
    print("=" * 70)

    # Test vector that doesn't need normalization
    normal_vec = [0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.3, 0.1]  # sum = 1.5
    normalized = normalize_vector(normal_vec, max_sum=1.5)
    assert normalized == normal_vec
    print(f"✅ Vector within limit unchanged: sum={sum(normalized):.2f}")

    # Test vector that needs normalization
    large_vec = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.8, 0.7]  # sum = 6.5
    normalized = normalize_vector(large_vec, max_sum=1.5)
    assert sum(normalized) <= 1.5 + 0.01  # Allow small floating point error

    # Check proportions are maintained
    ratio_original = large_vec[0] / large_vec[1]
    ratio_normalized = normalized[0] / normalized[1]
    assert abs(ratio_original - ratio_normalized) < 0.001

    print(f"✅ Oversized vector normalized: {sum(large_vec):.2f} → {sum(normalized):.2f}")

    return True


def test_get_all_presets():
    """Test get_all_presets returns copy."""
    print("\n" + "=" * 70)
    print("TEST: get_all_presets()")
    print("=" * 70)

    presets = get_all_presets()

    assert len(presets) == 11
    assert "neutral" in presets
    assert "happy" in presets

    # Verify it's a copy (modifying shouldn't affect original)
    original_len = len(EMOTION_PRESETS)
    presets["test"] = EmotionPreset("test", "Test", "🧪", "Test", [0]*8)
    assert len(EMOTION_PRESETS) == original_len

    print(f"✅ get_all_presets() returns {len(presets)} presets (copy)")
    return True


def test_get_preset_description():
    """Test preset description retrieval."""
    print("\n" + "=" * 70)
    print("TEST: get_preset_description()")
    print("=" * 70)

    # Test valid preset
    desc = get_preset_description("happy")
    assert "Happy" in desc
    assert len(desc) > 0
    print(f"✅ Happy description: {desc}")

    # Test invalid preset
    desc_invalid = get_preset_description("nonexistent")
    assert "Custom" in desc_invalid
    print(f"✅ Invalid preset returns custom message: {desc_invalid}")

    return True


def test_preset_semantic_correctness():
    """Test that preset vectors make semantic sense."""
    print("\n" + "=" * 70)
    print("TEST: Preset Semantic Correctness")
    print("=" * 70)

    # Happy should have high happiness (index 0)
    happy = get_preset_vector("happy")
    assert happy[0] > 0.5, "Happy should have high happiness"
    print(f"✅ Happy has high happiness: {happy[0]}")

    # Angry should have high anger (index 1)
    angry = get_preset_vector("angry")
    assert angry[1] > 0.5, "Angry should have high anger"
    print(f"✅ Angry has high anger: {angry[1]}")

    # Sad should have high sadness (index 2)
    sad = get_preset_vector("sad")
    assert sad[2] > 0.5, "Sad should have high sadness"
    print(f"✅ Sad has high sadness: {sad[2]}")

    # Surprised should have high surprise (index 3)
    surprised = get_preset_vector("surprised")
    assert surprised[3] > 0.5, "Surprised should have high surprise"
    print(f"✅ Surprised has high surprise: {surprised[3]}")

    # Disgusted should have high disgust (index 4)
    disgusted = get_preset_vector("disgusted")
    assert disgusted[4] > 0.5, "Disgusted should have high disgust"
    print(f"✅ Disgusted has high disgust: {disgusted[4]}")

    # Fearful should have high fear (index 5)
    fearful = get_preset_vector("fearful")
    assert fearful[5] > 0.5, "Fearful should have high fear"
    print(f"✅ Fearful has high fear: {fearful[5]}")

    # Excited should have high arousal (index 6)
    excited = get_preset_vector("excited")
    assert excited[6] > 0.5, "Excited should have high arousal"
    print(f"✅ Excited has high arousal: {excited[6]}")

    # Calm should have high calm (index 7)
    calm = get_preset_vector("calm")
    assert calm[7] > 0.5, "Calm should have high calm"
    print(f"✅ Calm has high calm: {calm[7]}")

    return True


def main():
    """Run all tests."""
    print("\n🧪 Testing Emotion Presets\n")

    tests = [
        ("EmotionPreset Dataclass", test_emotion_preset_dataclass),
        ("Preset Definitions", test_preset_definitions),
        ("get_preset_choices()", test_get_preset_choices),
        ("get_preset_vector()", test_get_preset_vector),
        ("mix_emotions()", test_mix_emotions),
        ("normalize_vector()", test_normalize_vector),
        ("get_all_presets()", test_get_all_presets),
        ("get_preset_description()", test_get_preset_description),
        ("Semantic Correctness", test_preset_semantic_correctness),
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
