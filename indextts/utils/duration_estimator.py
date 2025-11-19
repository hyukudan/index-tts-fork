"""
Duration Estimation for IndexTTS-2

Provides duration estimation and speech rate analysis for TTS generation.
"""

import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DurationEstimate:
    """Duration estimation result."""
    estimated_seconds: float
    min_seconds: float
    max_seconds: float
    character_count: int
    word_count: int
    speech_rate: str  # "slow", "normal", "fast"
    confidence: str  # "low", "medium", "high"


# Language-specific speech rates (characters per second)
# Based on average speaking speeds
SPEECH_RATES = {
    "zh": {  # Chinese
        "slow": 3.5,
        "normal": 5.0,
        "fast": 7.0,
    },
    "en": {  # English
        "slow": 10.0,
        "normal": 14.0,
        "fast": 18.0,
    },
    "ja": {  # Japanese
        "slow": 4.0,
        "normal": 6.0,
        "fast": 8.5,
    },
    "default": {  # Fallback
        "slow": 8.0,
        "normal": 12.0,
        "fast": 16.0,
    }
}

# Punctuation pause times (seconds)
PAUSE_TIMES = {
    '.': 0.6,
    '!': 0.6,
    '?': 0.6,
    ',': 0.3,
    ';': 0.4,
    ':': 0.4,
    '。': 0.6,  # Chinese period
    '！': 0.6,
    '？': 0.6,
    '，': 0.3,
    '、': 0.2,
}


def detect_language(text: str) -> str:
    """
    Detect primary language of text.

    Args:
        text: Input text

    Returns:
        Language code ('zh', 'en', 'ja', or 'default')
    """
    # Count characters by type
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    ascii_chars = len(re.findall(r'[a-zA-Z]', text))

    total_chars = chinese_chars + japanese_chars + ascii_chars

    if total_chars == 0:
        return "default"

    # Determine primary language
    if chinese_chars / total_chars > 0.3:
        return "zh"
    elif japanese_chars / total_chars > 0.3:
        return "ja"
    elif ascii_chars / total_chars > 0.5:
        return "en"
    else:
        return "default"


def count_words(text: str, lang: str) -> int:
    """
    Count words in text, language-aware.

    Args:
        text: Input text
        lang: Language code

    Returns:
        Word count
    """
    if lang in ["zh", "ja"]:
        # CJK languages: each character is roughly a word/morpheme
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
        # Also count ASCII words
        ascii_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        return cjk_chars + ascii_words
    else:
        # Western languages: space-separated words
        words = re.findall(r'\b\w+\b', text)
        return len(words)


def estimate_pause_time(text: str) -> float:
    """
    Estimate total pause time from punctuation.

    Args:
        text: Input text

    Returns:
        Total pause time in seconds
    """
    total_pause = 0.0

    for char in text:
        pause = PAUSE_TIMES.get(char, 0.0)
        total_pause += pause

    return total_pause


def estimate_duration(
    text: str,
    speech_rate: str = "normal",
    lang: Optional[str] = None
) -> DurationEstimate:
    """
    Estimate speech duration for given text.

    Args:
        text: Input text to synthesize
        speech_rate: Speech rate ("slow", "normal", "fast")
        lang: Optional language code (auto-detected if None)

    Returns:
        DurationEstimate with timing information
    """
    # Detect language if not provided
    if lang is None:
        lang = detect_language(text)

    # Get speech rates for language
    rates = SPEECH_RATES.get(lang, SPEECH_RATES["default"])
    chars_per_second = rates.get(speech_rate, rates["normal"])

    # Count characters (excluding whitespace and punctuation)
    text_chars = re.sub(r'[\s\p{P}]', '', text)
    char_count = len(text_chars)

    # Count words
    word_count = count_words(text, lang)

    # Base duration from character count
    base_duration = char_count / chars_per_second

    # Add pause time
    pause_time = estimate_pause_time(text)

    # Total estimated duration
    estimated = base_duration + pause_time

    # Calculate confidence based on text length
    if char_count < 20:
        confidence = "low"
        variance = 0.5  # ±50%
    elif char_count < 100:
        confidence = "medium"
        variance = 0.3  # ±30%
    else:
        confidence = "high"
        variance = 0.2  # ±20%

    min_duration = estimated * (1.0 - variance)
    max_duration = estimated * (1.0 + variance)

    return DurationEstimate(
        estimated_seconds=estimated,
        min_seconds=min_duration,
        max_seconds=max_duration,
        character_count=char_count,
        word_count=word_count,
        speech_rate=speech_rate,
        confidence=confidence
    )


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2.5s" or "1m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"


def get_duration_display(text: str, speech_rate: str = "normal") -> str:
    """
    Get formatted duration display for UI.

    Args:
        text: Input text
        speech_rate: Speech rate

    Returns:
        Formatted display string
    """
    if not text or len(text.strip()) == 0:
        return "📊 No text to estimate"

    estimate = estimate_duration(text, speech_rate)

    lines = [
        f"📊 **Duration Estimate**",
        f"- Characters: {estimate.character_count}",
        f"- Words/Morphemes: {estimate.word_count}",
        f"- Estimated: {format_duration(estimate.estimated_seconds)}",
        f"- Range: {format_duration(estimate.min_seconds)} - {format_duration(estimate.max_seconds)}",
        f"- Speed: {estimate.speech_rate.capitalize()}",
        f"- Confidence: {estimate.confidence.capitalize()}"
    ]

    return "\n".join(lines)


def calculate_speed_multiplier(target_duration: float, estimated_duration: float) -> float:
    """
    Calculate speed multiplier needed to hit target duration.

    Args:
        target_duration: Desired duration in seconds
        estimated_duration: Estimated duration at normal speed

    Returns:
        Speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
    """
    if estimated_duration == 0 or target_duration == 0:
        return 1.0

    return estimated_duration / target_duration
