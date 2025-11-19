"""
Model Comparison Module for IndexTTS.

Enables side-by-side comparison of different TTS models with the same input,
collecting performance metrics and generating outputs for A/B testing.
"""

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import librosa

from indextts.utils.model_manager import ModelManager, ModelMetadata


logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Performance metrics for a single generation."""
    model_name: str
    model_version: str
    gpt_time: float  # Seconds
    s2mel_time: float  # Seconds
    vocoder_time: float  # Seconds
    total_time: float  # Seconds
    audio_duration: float  # Seconds
    rtf: float  # Real-time factor (total_time / audio_duration)
    vram_peak_gb: float  # Peak VRAM usage
    vram_allocated_gb: float  # Allocated VRAM after generation
    gpu_device: str  # GPU used
    precision: str  # FP16 or FP32
    text_length: int  # Number of characters
    audio_sample_rate: int
    audio_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of comparing two models."""
    model_a_metrics: GenerationMetrics
    model_b_metrics: GenerationMetrics
    audio_a_path: str
    audio_b_path: str
    prompt_text: str
    prompt_audio_path: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_a': self.model_a_metrics.to_dict(),
            'model_b': self.model_b_metrics.to_dict(),
            'audio_a': self.audio_a_path,
            'audio_b': self.audio_b_path,
            'prompt_text': self.prompt_text,
            'prompt_audio': self.prompt_audio_path,
            'timestamp': self.timestamp
        }


class ModelComparator:
    """
    Handles side-by-side comparison of TTS models.

    Uses ModelManager for hot-swapping between models to generate
    outputs with identical inputs and collect performance metrics.
    """

    def __init__(self, model_manager: ModelManager):
        """
        Initialize ModelComparator.

        Args:
            model_manager: ModelManager instance for loading models
        """
        self.model_manager = model_manager

    def _collect_metrics(
        self,
        model,
        metadata: ModelMetadata,
        text_length: int,
        audio_path: str,
        gpu_device: str,
        use_fp16: bool
    ) -> GenerationMetrics:
        """
        Collect performance metrics from a model instance after generation.

        Args:
            model: IndexTTS2 instance
            metadata: Model metadata
            text_length: Length of input text
            audio_path: Path to generated audio
            gpu_device: GPU device string
            use_fp16: Whether FP16 was used

        Returns:
            GenerationMetrics object
        """
        # Get timing info from model (if available)
        gpt_time = getattr(model, 'last_gpt_time', 0.0)
        s2mel_time = getattr(model, 'last_s2mel_time', 0.0)
        vocoder_time = getattr(model, 'last_bigvgan_time', 0.0)
        total_time = getattr(model, 'last_total_time', gpt_time + s2mel_time + vocoder_time)
        audio_duration = getattr(model, 'last_audio_duration', 0.0)
        rtf = getattr(model, 'last_rtf', total_time / audio_duration if audio_duration > 0 else 0.0)

        # Get memory info
        mem_info = self.model_manager.get_memory_usage()
        vram_allocated = mem_info.get('allocated_gb', 0.0)
        vram_peak = mem_info.get('reserved_gb', 0.0)

        # Load audio to get actual duration and sample rate
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            actual_duration = len(audio) / sr
            sample_rate = sr
        except:
            actual_duration = audio_duration
            sample_rate = 22050

        return GenerationMetrics(
            model_name=metadata.filename,
            model_version=metadata.version,
            gpt_time=gpt_time,
            s2mel_time=s2mel_time,
            vocoder_time=vocoder_time,
            total_time=total_time,
            audio_duration=actual_duration,
            rtf=rtf,
            vram_peak_gb=vram_peak,
            vram_allocated_gb=vram_allocated,
            gpu_device=gpu_device,
            precision="FP16" if use_fp16 else "FP32",
            text_length=text_length,
            audio_sample_rate=sample_rate,
            audio_path=audio_path
        )

    def compare_models(
        self,
        model_a_path: str,
        model_b_path: str,
        text: str,
        prompt_audio: str,
        output_dir: str,
        gpu_id: int = 0,
        use_fp16: bool = False,
        **generation_kwargs
    ) -> ComparisonResult:
        """
        Generate audio with two different models using identical inputs.

        Args:
            model_a_path: Path to first model checkpoint
            model_b_path: Path to second model checkpoint
            text: Input text for TTS
            prompt_audio: Path to voice prompt audio
            output_dir: Directory to save generated audio files
            gpu_id: GPU device ID
            use_fp16: Use FP16 precision
            **generation_kwargs: Additional arguments for generation

        Returns:
            ComparisonResult with metrics and output paths
        """
        from datetime import datetime
        import os

        logger.info(f"Starting model comparison")
        logger.info(f"  Model A: {Path(model_a_path).name}")
        logger.info(f"  Model B: {Path(model_b_path).name}")
        logger.info(f"  Text: {text[:50]}...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate with Model A
        logger.info("Generating with Model A...")
        model_a = self.model_manager.load_model(
            model_a_path,
            gpu_id=gpu_id,
            use_fp16=use_fp16
        )
        metadata_a = self.model_manager.get_current_metadata()

        output_a_path = str(output_path / f"compare_a_{timestamp}.wav")

        try:
            model_a.infer(
                spk_audio_prompt=prompt_audio,
                text=text,
                output_path=output_a_path,
                verbose=False,
                **generation_kwargs
            )

            metrics_a = self._collect_metrics(
                model_a,
                metadata_a,
                len(text),
                output_a_path,
                f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu",
                use_fp16
            )

            logger.info(f"  Model A RTF: {metrics_a.rtf:.3f}")

        except Exception as e:
            logger.error(f"Model A generation failed: {e}")
            raise

        # Unload Model A before loading Model B
        self.model_manager.unload_current_model()

        # Generate with Model B
        logger.info("Generating with Model B...")
        model_b = self.model_manager.load_model(
            model_b_path,
            gpu_id=gpu_id,
            use_fp16=use_fp16
        )
        metadata_b = self.model_manager.get_current_metadata()

        output_b_path = str(output_path / f"compare_b_{timestamp}.wav")

        try:
            model_b.infer(
                spk_audio_prompt=prompt_audio,
                text=text,
                output_path=output_b_path,
                verbose=False,
                **generation_kwargs
            )

            metrics_b = self._collect_metrics(
                model_b,
                metadata_b,
                len(text),
                output_b_path,
                f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu",
                use_fp16
            )

            logger.info(f"  Model B RTF: {metrics_b.rtf:.3f}")

        except Exception as e:
            logger.error(f"Model B generation failed: {e}")
            raise

        # Create comparison result
        result = ComparisonResult(
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            audio_a_path=output_a_path,
            audio_b_path=output_b_path,
            prompt_text=text,
            prompt_audio_path=prompt_audio,
            timestamp=timestamp
        )

        logger.info("Comparison completed successfully")

        return result

    def generate_waveform_comparison(
        self,
        audio_a_path: str,
        audio_b_path: str,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ) -> Optional[str]:
        """
        Generate a visual comparison of waveforms from two audio files.

        Args:
            audio_a_path: Path to first audio file
            audio_b_path: Path to second audio file
            output_path: Path to save comparison image (PNG)
            figsize: Figure size (width, height)

        Returns:
            Path to saved image, or None if matplotlib not available
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            logger.warning("matplotlib not available, skipping waveform visualization")
            return None

        # Load audio files
        audio_a, sr_a = librosa.load(audio_a_path, sr=None)
        audio_b, sr_b = librosa.load(audio_b_path, sr=None)

        # Resample if needed
        if sr_a != sr_b:
            logger.warning(f"Sample rate mismatch: {sr_a} vs {sr_b}, resampling to 22050")
            audio_a = librosa.resample(audio_a, orig_sr=sr_a, target_sr=22050)
            audio_b = librosa.resample(audio_b, orig_sr=sr_b, target_sr=22050)
            sr = 22050
        else:
            sr = sr_a

        # Create time axes
        time_a = np.arange(len(audio_a)) / sr
        time_b = np.arange(len(audio_b)) / sr

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot Model A
        axes[0].plot(time_a, audio_a, color='#1f77b4', linewidth=0.5)
        axes[0].set_ylabel('Amplitude', fontsize=10)
        axes[0].set_title(f'Model A ({Path(audio_a_path).stem})', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-1, 1)

        # Plot Model B
        axes[1].plot(time_b, audio_b, color='#ff7f0e', linewidth=0.5)
        axes[1].set_ylabel('Amplitude', fontsize=10)
        axes[1].set_xlabel('Time (s)', fontsize=10)
        axes[1].set_title(f'Model B ({Path(audio_b_path).stem})', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-1, 1)

        plt.tight_layout()

        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Waveform comparison saved to {output_path}")
            plt.close()
            return output_path

        return None

    def format_metrics_table(self, result: ComparisonResult) -> str:
        """
        Format comparison metrics as a readable table.

        Args:
            result: ComparisonResult object

        Returns:
            Formatted string table
        """
        m_a = result.model_a_metrics
        m_b = result.model_b_metrics

        # Calculate winner for each metric (lower is better for times, higher for speed)
        rtf_winner = "A" if m_a.rtf < m_b.rtf else "B"
        time_winner = "A" if m_a.total_time < m_b.total_time else "B"
        vram_winner = "A" if m_a.vram_peak_gb < m_b.vram_peak_gb else "B"

        table = f"""
╔════════════════════════════════════════════════════════════════╗
║                   MODEL COMPARISON RESULTS                      ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  Metric              │  Model A         │  Model B         │ ✓ ║
║ ─────────────────────┼──────────────────┼──────────────────┼───║
║  Model Name          │ {m_a.model_name:16s} │ {m_b.model_name:16s} │   ║
║  Version             │ {m_a.model_version:16s} │ {m_b.model_version:16s} │   ║
║  Precision           │ {m_a.precision:16s} │ {m_b.precision:16s} │   ║
║                                                                 ║
║  GPT Time            │ {m_a.gpt_time:14.2f}s │ {m_b.gpt_time:14.2f}s │   ║
║  S2Mel Time          │ {m_a.s2mel_time:14.2f}s │ {m_b.s2mel_time:14.2f}s │   ║
║  Vocoder Time        │ {m_a.vocoder_time:14.2f}s │ {m_b.vocoder_time:14.2f}s │   ║
║  Total Time          │ {m_a.total_time:14.2f}s │ {m_b.total_time:14.2f}s │ {time_winner} ║
║                                                                 ║
║  Audio Duration      │ {m_a.audio_duration:14.2f}s │ {m_b.audio_duration:14.2f}s │   ║
║  RTF                 │ {m_a.rtf:16.3f} │ {m_b.rtf:16.3f} │ {rtf_winner} ║
║                                                                 ║
║  VRAM Peak           │ {m_a.vram_peak_gb:13.2f}GB │ {m_b.vram_peak_gb:13.2f}GB │ {vram_winner} ║
║  VRAM Allocated      │ {m_a.vram_allocated_gb:13.2f}GB │ {m_b.vram_allocated_gb:13.2f}GB │   ║
║                                                                 ║
╚════════════════════════════════════════════════════════════════╝
        """.strip()

        return table
