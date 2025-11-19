#!/usr/bin/env python3
"""
Test suite for Duration Estimator utility.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.utils.duration_estimator import (
    DurationEstimate,
    SPEECH_RATES,
    PAUSE_TIMES,
    detect_language,
    count_words,
    estimate_pause_time,
    estimate_duration,
    format_duration,
    get_duration_display,
    calculate_speed_multiplier
)


def test_duration_estimate_dataclass():
    """Test DurationEstimate dataclass."""
    print("=" * 70)
    print("TEST: DurationEstimate Dataclass")
    print("=" * 70)

    estimate = DurationEstimate(
        estimated_seconds=5.2,
        min_seconds=4.5,
        max_seconds=6.0,
        character_count=50,
        word_count=10,
        speech_rate="normal",
        confidence="high"
    )

    assert estimate.estimated_seconds == 5.2
    assert estimate.min_seconds == 4.5
    assert estimate.max_seconds == 6.0
    assert estimate.character_count == 50
    assert estimate.word_count == 10
    assert estimate.speech_rate == "normal"
    assert estimate.confidence == "high"

    print("✅ DurationEstimate dataclass works correctly")
    return True


def test_speech_rates_definition():
    """Test that speech rates are properly defined."""
    print("\n" + "=" * 70)
    print("TEST: Speech Rates Definition")
    print("=" * 70)

    required_langs = ["zh", "en", "ja", "default"]
    required_rates = ["slow", "normal", "fast"]

    for lang in required_langs:
        assert lang in SPEECH_RATES, f"Missing language: {lang}"

        for rate in required_rates:
            assert rate in SPEECH_RATES[lang], f"Missing rate {rate} for {lang}"
            value = SPEECH_RATES[lang][rate]
            assert value > 0, f"Invalid rate value for {lang}.{rate}: {value}"

    # Check that fast > normal > slow
    for lang in required_langs:
        assert SPEECH_RATES[lang]["fast"] > SPEECH_RATES[lang]["normal"]
        assert SPEECH_RATES[lang]["normal"] > SPEECH_RATES[lang]["slow"]

    print(f"✅ Speech rates properly defined for {len(required_langs)} languages")
    for lang in required_langs:
        rates = SPEECH_RATES[lang]
        print(f"   {lang}: slow={rates['slow']}, normal={rates['normal']}, fast={rates['fast']} chars/sec")

    return True


def test_pause_times_definition():
    """Test that pause times are defined."""
    print("\n" + "=" * 70)
    print("TEST: Pause Times Definition")
    print("=" * 70)

    required_punctuation = ['.', '!', '?', ',', ';', ':', '。', '！', '？', '，', '、']

    for punct in required_punctuation:
        assert punct in PAUSE_TIMES, f"Missing pause time for: {punct}"
        value = PAUSE_TIMES[punct]
        assert value >= 0, f"Invalid pause time for {punct}: {value}"

    print(f"✅ Pause times defined for {len(PAUSE_TIMES)} punctuation marks")
    return True


def test_detect_language():
    """Test language detection."""
    print("\n" + "=" * 70)
    print("TEST: detect_language()")
    print("=" * 70)

    # Test English
    lang_en = detect_language("Hello world, this is a test.")
    assert lang_en == "en", f"Expected 'en', got '{lang_en}'"
    print(f"✅ English detected: '{lang_en}'")

    # Test Chinese
    lang_zh = detect_language("你好世界，这是一个测试。")
    assert lang_zh == "zh", f"Expected 'zh', got '{lang_zh}'"
    print(f"✅ Chinese detected: '{lang_zh}'")

    # Test Japanese
    lang_ja = detect_language("こんにちは世界、これはテストです。")
    assert lang_ja == "ja", f"Expected 'ja', got '{lang_ja}'"
    print(f"✅ Japanese detected: '{lang_ja}'")

    # Test mixed (should detect primary language)
    lang_mixed = detect_language("Hello 你好 world")
    print(f"✅ Mixed text detected as: '{lang_mixed}'")

    # Test empty/no recognizable chars
    lang_empty = detect_language("123 !@# $%^")
    assert lang_empty == "default"
    print(f"✅ Empty/symbols detected as: '{lang_empty}'")

    return True


def test_count_words():
    """Test word counting for different languages."""
    print("\n" + "=" * 70)
    print("TEST: count_words()")
    print("=" * 70)

    # English: space-separated words
    count_en = count_words("Hello world this is a test", "en")
    assert count_en == 6, f"Expected 6 English words, got {count_en}"
    print(f"✅ English word count: {count_en}")

    # Chinese: each character counts
    count_zh = count_words("你好世界", "zh")
    assert count_zh == 4, f"Expected 4 Chinese characters, got {count_zh}"
    print(f"✅ Chinese character count: {count_zh}")

    # Japanese: each character counts
    count_ja = count_words("こんにちは", "ja")
    assert count_ja == 5, f"Expected 5 Japanese characters, got {count_ja}"
    print(f"✅ Japanese character count: {count_ja}")

    # Mixed CJK + English
    count_mixed = count_words("你好 hello 世界", "zh")
    # Should count: 你好世界 (4 chars) + hello (1 word) = 5
    assert count_mixed == 5, f"Expected 5, got {count_mixed}"
    print(f"✅ Mixed CJK+English count: {count_mixed}")

    return True


def test_estimate_pause_time():
    """Test pause time estimation from punctuation."""
    print("\n" + "=" * 70)
    print("TEST: estimate_pause_time()")
    print("=" * 70)

    # No punctuation
    pause_none = estimate_pause_time("Hello world")
    assert pause_none == 0.0
    print(f"✅ No punctuation: {pause_none}s")

    # One period
    pause_period = estimate_pause_time("Hello.")
    assert pause_period == PAUSE_TIMES['.']
    print(f"✅ One period: {pause_period}s")

    # Multiple punctuation
    text_multi = "Hello, world! How are you? I'm fine."
    pause_multi = estimate_pause_time(text_multi)
    expected = PAUSE_TIMES[','] + PAUSE_TIMES['!'] + PAUSE_TIMES['?'] + PAUSE_TIMES['.']
    assert abs(pause_multi - expected) < 0.001
    print(f"✅ Multiple punctuation: {pause_multi}s")

    # Chinese punctuation
    text_zh = "你好。世界！"
    pause_zh = estimate_pause_time(text_zh)
    expected_zh = PAUSE_TIMES['。'] + PAUSE_TIMES['！']
    assert abs(pause_zh - expected_zh) < 0.001
    print(f"✅ Chinese punctuation: {pause_zh}s")

    return True


def test_estimate_duration_english():
    """Test duration estimation for English text."""
    print("\n" + "=" * 70)
    print("TEST: estimate_duration() - English")
    print("=" * 70)

    text = "Hello world, this is a test sentence."

    # Test normal speed
    estimate = estimate_duration(text, speech_rate="normal")

    assert estimate.estimated_seconds > 0
    assert estimate.min_seconds < estimate.estimated_seconds
    assert estimate.estimated_seconds < estimate.max_seconds
    assert estimate.character_count > 0
    assert estimate.word_count > 0
    assert estimate.speech_rate == "normal"
    assert estimate.confidence in ["low", "medium", "high"]

    print(f"✅ English estimation:")
    print(f"   Text: '{text}'")
    print(f"   Characters: {estimate.character_count}")
    print(f"   Words: {estimate.word_count}")
    print(f"   Estimated: {estimate.estimated_seconds:.2f}s")
    print(f"   Range: {estimate.min_seconds:.2f}s - {estimate.max_seconds:.2f}s")
    print(f"   Confidence: {estimate.confidence}")

    # Test different speeds
    estimate_slow = estimate_duration(text, speech_rate="slow")
    estimate_fast = estimate_duration(text, speech_rate="fast")

    assert estimate_slow.estimated_seconds > estimate.estimated_seconds
    assert estimate_fast.estimated_seconds < estimate.estimated_seconds

    print(f"✅ Speed variations: slow={estimate_slow.estimated_seconds:.2f}s, "
          f"normal={estimate.estimated_seconds:.2f}s, fast={estimate_fast.estimated_seconds:.2f}s")

    return True


def test_estimate_duration_chinese():
    """Test duration estimation for Chinese text."""
    print("\n" + "=" * 70)
    print("TEST: estimate_duration() - Chinese")
    print("=" * 70)

    text = "你好世界，这是一个测试句子。"

    estimate = estimate_duration(text, speech_rate="normal")

    assert estimate.estimated_seconds > 0
    assert estimate.character_count > 0
    assert estimate.word_count > 0

    print(f"✅ Chinese estimation:")
    print(f"   Text: '{text}'")
    print(f"   Characters: {estimate.character_count}")
    print(f"   Words/Morphemes: {estimate.word_count}")
    print(f"   Estimated: {estimate.estimated_seconds:.2f}s")

    return True


def test_estimate_duration_confidence():
    """Test confidence levels based on text length."""
    print("\n" + "=" * 70)
    print("TEST: Confidence Levels")
    print("=" * 70)

    # Short text (< 20 chars) -> low confidence
    short = estimate_duration("Hi")
    assert short.confidence == "low"
    print(f"✅ Short text ({short.character_count} chars): {short.confidence} confidence")

    # Medium text (20-100 chars) -> medium confidence
    medium = estimate_duration("This is a medium length test sentence with enough words.")
    assert medium.confidence == "medium"
    print(f"✅ Medium text ({medium.character_count} chars): {medium.confidence} confidence")

    # Long text (> 100 chars) -> high confidence
    long_text = "This is a very long test sentence. " * 10
    long = estimate_duration(long_text)
    assert long.confidence == "high"
    print(f"✅ Long text ({long.character_count} chars): {long.confidence} confidence")

    return True


def test_format_duration():
    """Test duration formatting."""
    print("\n" + "=" * 70)
    print("TEST: format_duration()")
    print("=" * 70)

    # Short duration (< 60s)
    fmt_short = format_duration(5.7)
    assert "5.7s" in fmt_short
    print(f"✅ Short duration: {fmt_short}")

    # Medium duration (1-2 minutes)
    fmt_medium = format_duration(75.3)
    assert "1m" in fmt_medium
    assert "15s" in fmt_medium
    print(f"✅ Medium duration: {fmt_medium}")

    # Long duration (> 2 minutes)
    fmt_long = format_duration(150.8)
    assert "2m" in fmt_long
    assert "31s" in fmt_long
    print(f"✅ Long duration: {fmt_long}")

    return True


def test_get_duration_display():
    """Test UI display string generation."""
    print("\n" + "=" * 70)
    print("TEST: get_duration_display()")
    print("=" * 70)

    # Empty text
    display_empty = get_duration_display("")
    assert "No text" in display_empty
    print(f"✅ Empty text display: First line = '{display_empty.split(chr(10))[0]}'")

    # Normal text
    text = "Hello world, this is a test."
    display = get_duration_display(text, speech_rate="normal")

    assert "Duration Estimate" in display
    assert "Characters:" in display
    assert "Words" in display
    assert "Estimated:" in display
    assert "Range:" in display
    assert "Speed:" in display
    assert "Confidence:" in display

    print(f"✅ Normal text display:")
    for line in display.split('\n')[:3]:
        print(f"   {line}")

    return True


def test_calculate_speed_multiplier():
    """Test speed multiplier calculation."""
    print("\n" + "=" * 70)
    print("TEST: calculate_speed_multiplier()")
    print("=" * 70)

    # Target = estimated -> multiplier = 1.0
    mult_equal = calculate_speed_multiplier(target_duration=5.0, estimated_duration=5.0)
    assert abs(mult_equal - 1.0) < 0.01
    print(f"✅ Equal durations: multiplier = {mult_equal}")

    # Target < estimated -> multiplier > 1.0 (speed up)
    mult_faster = calculate_speed_multiplier(target_duration=3.0, estimated_duration=5.0)
    assert mult_faster > 1.0
    print(f"✅ Need faster (5s → 3s): multiplier = {mult_faster:.2f}")

    # Target > estimated -> multiplier < 1.0 (slow down)
    mult_slower = calculate_speed_multiplier(target_duration=7.0, estimated_duration=5.0)
    assert mult_slower < 1.0
    print(f"✅ Need slower (5s → 7s): multiplier = {mult_slower:.2f}")

    # Edge case: zero durations
    mult_zero = calculate_speed_multiplier(target_duration=0, estimated_duration=5.0)
    assert mult_zero == 1.0
    print(f"✅ Zero target: multiplier = {mult_zero}")

    return True


def test_language_auto_detection():
    """Test automatic language detection in estimate_duration."""
    print("\n" + "=" * 70)
    print("TEST: Auto Language Detection")
    print("=" * 70)

    # English (no lang specified)
    est_en = estimate_duration("Hello world", speech_rate="normal", lang=None)
    print(f"✅ Auto-detected English: {est_en.estimated_seconds:.2f}s")

    # Chinese (no lang specified)
    est_zh = estimate_duration("你好世界", speech_rate="normal", lang=None)
    print(f"✅ Auto-detected Chinese: {est_zh.estimated_seconds:.2f}s")

    # Explicit language override
    est_explicit = estimate_duration("Hello", speech_rate="normal", lang="zh")
    print(f"✅ Explicit Chinese (English text): {est_explicit.estimated_seconds:.2f}s")

    return True


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 70)
    print("TEST: Edge Cases")
    print("=" * 70)

    # Very short text
    est_short = estimate_duration("Hi")
    assert est_short.estimated_seconds > 0
    assert est_short.confidence == "low"
    print(f"✅ Very short text: {est_short.estimated_seconds:.2f}s, confidence={est_short.confidence}")

    # Only punctuation
    est_punct = estimate_duration("... !!! ???")
    assert est_punct.estimated_seconds > 0
    print(f"✅ Only punctuation: {est_punct.estimated_seconds:.2f}s")

    # Only spaces
    est_spaces = estimate_duration("     ")
    assert est_spaces.character_count == 0
    print(f"✅ Only spaces: {est_spaces.character_count} chars")

    # Very long text
    long_text = "This is a test sentence. " * 100
    est_long = estimate_duration(long_text)
    assert est_long.confidence == "high"
    assert est_long.estimated_seconds > 10
    print(f"✅ Very long text: {est_long.estimated_seconds:.2f}s, confidence={est_long.confidence}")

    return True


def main():
    """Run all tests."""
    print("\n🧪 Testing Duration Estimator\n")

    tests = [
        ("DurationEstimate Dataclass", test_duration_estimate_dataclass),
        ("Speech Rates Definition", test_speech_rates_definition),
        ("Pause Times Definition", test_pause_times_definition),
        ("detect_language()", test_detect_language),
        ("count_words()", test_count_words),
        ("estimate_pause_time()", test_estimate_pause_time),
        ("estimate_duration() - English", test_estimate_duration_english),
        ("estimate_duration() - Chinese", test_estimate_duration_chinese),
        ("Confidence Levels", test_estimate_duration_confidence),
        ("format_duration()", test_format_duration),
        ("get_duration_display()", test_get_duration_display),
        ("calculate_speed_multiplier()", test_calculate_speed_multiplier),
        ("Auto Language Detection", test_language_auto_detection),
        ("Edge Cases", test_edge_cases),
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
