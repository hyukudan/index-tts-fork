"""
Emotion Presets for IndexTTS-2

Provides predefined emotion vectors and mixing capabilities for easy emotional control.
Based on the 8-dimensional emotion space of IndexTTS2.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EmotionPreset:
    """Preset emotion configuration."""
    name: str
    display_name: str
    emoji: str
    description: str
    vector: List[float]  # 8-dimensional: [happiness, anger, sadness, surprise, disgust, fear, arousal, calm]


# Standard emotion presets based on IndexTTS2's 8-dimensional emotion space
EMOTION_PRESETS = {
    "neutral": EmotionPreset(
        name="neutral",
        display_name="Neutral",
        emoji="😐",
        description="Balanced, neutral emotion",
        vector=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]  # Slight calm
    ),
    "happy": EmotionPreset(
        name="happy",
        display_name="Happy & Upbeat",
        emoji="😊",
        description="Joyful and positive",
        vector=[1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.6, 0.0]  # High happiness, some surprise, medium arousal
    ),
    "excited": EmotionPreset(
        name="excited",
        display_name="Excited & Energetic",
        emoji="🎉",
        description="High energy and enthusiasm",
        vector=[0.8, 0.0, 0.0, 0.6, 0.0, 0.0, 1.0, 0.0]  # High arousal, happiness, surprise
    ),
    "sad": EmotionPreset(
        name="sad",
        display_name="Sad & Melancholic",
        emoji="😢",
        description="Sorrowful and downcast",
        vector=[0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.3]  # High sadness, low arousal
    ),
    "angry": EmotionPreset(
        name="angry",
        display_name="Angry & Intense",
        emoji="😠",
        description="Frustrated or irritated",
        vector=[0.0, 1.0, 0.0, 0.0, 0.3, 0.0, 0.8, 0.0]  # High anger, some disgust, high arousal
    ),
    "fearful": EmotionPreset(
        name="fearful",
        display_name="Fearful & Anxious",
        emoji="😰",
        description="Worried or scared",
        vector=[0.0, 0.0, 0.2, 0.3, 0.0, 0.9, 0.6, 0.0]  # High fear, some surprise, medium arousal
    ),
    "calm": EmotionPreset(
        name="calm",
        display_name="Calm & Peaceful",
        emoji="😌",
        description="Relaxed and tranquil",
        vector=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # High calm, slight happiness
    ),
    "surprised": EmotionPreset(
        name="surprised",
        display_name="Surprised & Amazed",
        emoji="😲",
        description="Astonished or shocked",
        vector=[0.3, 0.0, 0.0, 1.0, 0.0, 0.1, 0.7, 0.0]  # High surprise, medium arousal
    ),
    "disgusted": EmotionPreset(
        name="disgusted",
        display_name="Disgusted & Repulsed",
        emoji="🤢",
        description="Feeling revulsion",
        vector=[0.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.4, 0.0]  # High disgust, some anger
    ),
    "confident": EmotionPreset(
        name="confident",
        display_name="Confident & Assertive",
        emoji="💪",
        description="Self-assured and strong",
        vector=[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.2]  # High happiness, high arousal, slight calm
    ),
    "tender": EmotionPreset(
        name="tender",
        display_name="Tender & Gentle",
        emoji="🥰",
        description="Soft and affectionate",
        vector=[0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8]  # High happiness and calm, low arousal
    ),
}


def get_preset_choices() -> List[Tuple[str, str]]:
    """
    Get list of preset choices for Gradio dropdown.

    Returns:
        List of (display_label, preset_id) tuples
    """
    choices = []
    for preset_id, preset in EMOTION_PRESETS.items():
        label = f"{preset.emoji} {preset.display_name}"
        choices.append((label, preset_id))

    # Add custom option
    choices.append(("✨ Custom Mix...", "custom"))

    return choices


def get_preset_vector(preset_id: str) -> Optional[List[float]]:
    """
    Get emotion vector for a preset.

    Args:
        preset_id: ID of the preset

    Returns:
        8-dimensional emotion vector, or None if preset not found
    """
    preset = EMOTION_PRESETS.get(preset_id)
    return preset.vector if preset else None


def mix_emotions(preset_a: str, preset_b: str, ratio: float) -> List[float]:
    """
    Mix two emotion presets with a given ratio.

    Args:
        preset_a: First preset ID
        preset_b: Second preset ID
        ratio: Mix ratio (0.0 = 100% A, 1.0 = 100% B)

    Returns:
        Mixed 8-dimensional emotion vector
    """
    vec_a = get_preset_vector(preset_a)
    vec_b = get_preset_vector(preset_b)

    if vec_a is None or vec_b is None:
        # Fallback to neutral if preset not found
        return EMOTION_PRESETS["neutral"].vector.copy()

    # Linear interpolation
    mixed = []
    for i in range(8):
        value = vec_a[i] * (1.0 - ratio) + vec_b[i] * ratio
        mixed.append(value)

    return mixed


def normalize_vector(vector: List[float], max_sum: float = 1.5) -> List[float]:
    """
    Normalize emotion vector to ensure sum doesn't exceed max_sum.

    Args:
        vector: Input emotion vector
        max_sum: Maximum allowed sum

    Returns:
        Normalized vector
    """
    current_sum = sum(vector)
    if current_sum <= max_sum:
        return vector

    # Scale down proportionally
    scale = max_sum / current_sum
    return [v * scale for v in vector]


def get_all_presets() -> Dict[str, EmotionPreset]:
    """Get dictionary of all emotion presets."""
    return EMOTION_PRESETS.copy()


def get_preset_description(preset_id: str) -> str:
    """Get description for a preset."""
    preset = EMOTION_PRESETS.get(preset_id)
    if preset:
        return f"{preset.emoji} **{preset.display_name}**: {preset.description}"
    return "Custom emotion mix"
