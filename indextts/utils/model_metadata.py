"""
Model metadata extraction utilities for IndexTTS.
Provides information about GPT checkpoints and BPE tokenizers.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import sentencepiece as spm
    SPM_AVAILABLE = True
except ImportError:
    SPM_AVAILABLE = False


@dataclass
class ModelInfo:
    """Information about a GPT checkpoint."""
    path: str
    filename: str
    size_mb: float
    modified_date: datetime
    vocab_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    format: str = "unknown"  # "pytorch", "safetensors", etc.


@dataclass
class TokenizerInfo:
    """Information about a BPE tokenizer."""
    path: str
    filename: str
    size_mb: float
    modified_date: datetime
    vocab_size: Optional[int] = None
    model_type: Optional[str] = None  # "bpe", "unigram", etc.
    languages: List[str] = None  # Detected language support


class ModelMetadataExtractor:
    """Extract metadata from model checkpoints and tokenizers."""

    @staticmethod
    def get_gpt_info(checkpoint_path: str) -> Optional[ModelInfo]:
        """
        Extract information from a GPT checkpoint.

        Args:
            checkpoint_path: Path to .pth or .safetensors file

        Returns:
            ModelInfo object or None if extraction fails
        """
        path_obj = Path(checkpoint_path)
        if not path_obj.exists():
            return None

        # Get file stats
        stats = path_obj.stat()
        size_mb = stats.st_size / (1024 ** 2)
        modified_date = datetime.fromtimestamp(stats.st_mtime)

        info = ModelInfo(
            path=str(path_obj),
            filename=path_obj.name,
            size_mb=size_mb,
            modified_date=modified_date,
            format=path_obj.suffix[1:]  # Remove the dot
        )

        # Try to extract model architecture info
        if TORCH_AVAILABLE and checkpoint_path.endswith('.pth'):
            try:
                # Load checkpoint header only (not full model)
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location='cpu',
                    weights_only=False
                )

                # Extract vocab size from embedding layer
                if 'text_embedding.weight' in checkpoint:
                    info.vocab_size = checkpoint['text_embedding.weight'].shape[0]
                    info.embedding_dim = checkpoint['text_embedding.weight'].shape[1]
                elif 'model' in checkpoint and 'text_embedding.weight' in checkpoint['model']:
                    info.vocab_size = checkpoint['model']['text_embedding.weight'].shape[0]
                    info.embedding_dim = checkpoint['model']['text_embedding.weight'].shape[1]

                # Try to detect number of layers
                layer_keys = [k for k in checkpoint.keys() if 'gpt.h.' in k or 'transformer.h.' in k]
                if layer_keys:
                    # Extract layer numbers
                    layer_nums = []
                    for key in layer_keys:
                        try:
                            if 'gpt.h.' in key:
                                num = int(key.split('gpt.h.')[1].split('.')[0])
                            elif 'transformer.h.' in key:
                                num = int(key.split('transformer.h.')[1].split('.')[0])
                            else:
                                continue
                            layer_nums.append(num)
                        except (ValueError, IndexError):
                            continue
                    if layer_nums:
                        info.num_layers = max(layer_nums) + 1

                # Clean up
                del checkpoint

            except Exception as e:
                # If loading fails, return basic info
                pass

        return info

    @staticmethod
    def get_tokenizer_info(tokenizer_path: str) -> Optional[TokenizerInfo]:
        """
        Extract information from a BPE tokenizer.

        Args:
            tokenizer_path: Path to .model file

        Returns:
            TokenizerInfo object or None if extraction fails
        """
        path_obj = Path(tokenizer_path)
        if not path_obj.exists():
            return None

        # Get file stats
        stats = path_obj.stat()
        size_mb = stats.st_size / (1024 ** 2)
        modified_date = datetime.fromtimestamp(stats.st_mtime)

        info = TokenizerInfo(
            path=str(path_obj),
            filename=path_obj.name,
            size_mb=size_mb,
            modified_date=modified_date,
            languages=[]
        )

        # Try to extract tokenizer metadata
        if SPM_AVAILABLE:
            try:
                sp = spm.SentencePieceProcessor()
                sp.Load(tokenizer_path)

                info.vocab_size = sp.GetPieceSize()

                # Detect model type (not directly available, infer from name)
                if 'bpe' in path_obj.stem.lower():
                    info.model_type = 'bpe'
                elif 'unigram' in path_obj.stem.lower():
                    info.model_type = 'unigram'
                else:
                    info.model_type = 'unknown'

                # Try to detect language support by sampling vocab
                vocab_sample = [sp.IdToPiece(i) for i in range(min(100, info.vocab_size))]

                has_chinese = any('\u4e00' <= c <= '\u9fff' for piece in vocab_sample for c in piece)
                has_japanese = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for piece in vocab_sample for c in piece)
                has_korean = any('\uac00' <= c <= '\ud7af' for piece in vocab_sample for c in piece)

                if has_chinese:
                    info.languages.append('Chinese')
                if has_japanese:
                    info.languages.append('Japanese')
                if has_korean:
                    info.languages.append('Korean')

                # English is assumed if there are ASCII characters
                if any(piece.isascii() for piece in vocab_sample):
                    info.languages.append('English')

                # If no languages detected, mark as multilingual
                if not info.languages:
                    info.languages.append('Multilingual')

            except Exception:
                # If loading fails, return basic info
                pass

        return info

    @staticmethod
    def format_model_info(info: ModelInfo, detailed: bool = False) -> str:
        """
        Format model information as a human-readable string.

        Args:
            info: ModelInfo object
            detailed: Include detailed technical info

        Returns:
            Formatted string
        """
        if detailed:
            lines = [
                f"📄 {info.filename}",
                f"   Size: {info.size_mb:.1f} MB",
                f"   Modified: {info.modified_date.strftime('%Y-%m-%d %H:%M')}",
            ]

            if info.vocab_size:
                lines.append(f"   Vocab size: {info.vocab_size:,}")
            if info.embedding_dim:
                lines.append(f"   Embedding dim: {info.embedding_dim}")
            if info.num_layers:
                lines.append(f"   Layers: {info.num_layers}")

            return "\n".join(lines)
        else:
            parts = [info.filename, f"{info.size_mb:.1f}MB"]
            if info.vocab_size:
                parts.append(f"vocab:{info.vocab_size}")
            return " | ".join(parts)

    @staticmethod
    def format_tokenizer_info(info: TokenizerInfo, detailed: bool = False) -> str:
        """
        Format tokenizer information as a human-readable string.

        Args:
            info: TokenizerInfo object
            detailed: Include detailed technical info

        Returns:
            Formatted string
        """
        if detailed:
            lines = [
                f"📝 {info.filename}",
                f"   Size: {info.size_mb:.1f} MB",
                f"   Modified: {info.modified_date.strftime('%Y-%m-%d %H:%M')}",
            ]

            if info.vocab_size:
                lines.append(f"   Vocab size: {info.vocab_size:,}")
            if info.model_type:
                lines.append(f"   Type: {info.model_type}")
            if info.languages:
                lines.append(f"   Languages: {', '.join(info.languages)}")

            return "\n".join(lines)
        else:
            parts = [info.filename, f"{info.size_mb:.1f}MB"]
            if info.vocab_size:
                parts.append(f"vocab:{info.vocab_size}")
            if info.languages:
                parts.append(f"langs:{','.join(info.languages)}")
            return " | ".join(parts)

    @staticmethod
    def estimate_model_vram(checkpoint_path: str, dtype: str = "fp16") -> int:
        """
        Estimate VRAM usage for a model in MB.

        Args:
            checkpoint_path: Path to checkpoint
            dtype: Data type ("fp32", "fp16", "bf16")

        Returns:
            Estimated VRAM in MB (conservative)
        """
        path_obj = Path(checkpoint_path)
        if not path_obj.exists():
            return 0

        # Get file size as baseline
        file_size_mb = path_obj.stat().st_size / (1024 ** 2)

        # Multiply by overhead factor based on dtype
        # Model weights + activations + optimizer states + temp buffers
        overhead_factors = {
            "fp32": 2.5,  # Higher overhead for fp32
            "fp16": 2.0,
            "bf16": 2.0,
        }

        factor = overhead_factors.get(dtype, 2.0)
        estimated = int(file_size_mb * factor)

        return estimated

    @staticmethod
    def validate_model_compatibility(gpt_info: ModelInfo, tokenizer_info: TokenizerInfo) -> tuple[bool, str]:
        """
        Check if a GPT checkpoint and tokenizer are compatible.

        Args:
            gpt_info: ModelInfo for GPT checkpoint
            tokenizer_info: TokenizerInfo for tokenizer

        Returns:
            Tuple of (is_compatible, message)
        """
        if not gpt_info or not tokenizer_info:
            return False, "Missing model or tokenizer info"

        # Check vocab size match
        if gpt_info.vocab_size and tokenizer_info.vocab_size:
            if gpt_info.vocab_size != tokenizer_info.vocab_size:
                return False, (
                    f"Vocab size mismatch: GPT has {gpt_info.vocab_size}, "
                    f"tokenizer has {tokenizer_info.vocab_size}"
                )

        # All checks passed
        return True, "Compatible"


def get_gpt_info(checkpoint_path: str) -> Optional[ModelInfo]:
    """Extract information from a GPT checkpoint."""
    return ModelMetadataExtractor.get_gpt_info(checkpoint_path)


def get_tokenizer_info(tokenizer_path: str) -> Optional[TokenizerInfo]:
    """Extract information from a BPE tokenizer."""
    return ModelMetadataExtractor.get_tokenizer_info(tokenizer_path)


def format_model_display(checkpoint_path: str, tokenizer_path: Optional[str] = None) -> str:
    """
    Create a display-friendly string for model selection dropdown.

    Args:
        checkpoint_path: Path to GPT checkpoint
        tokenizer_path: Optional path to tokenizer

    Returns:
        Formatted string for display
    """
    gpt_info = get_gpt_info(checkpoint_path)
    if not gpt_info:
        return Path(checkpoint_path).name

    display = gpt_info.filename

    # Add size
    display += f" ({gpt_info.size_mb:.0f}MB"

    # Add vocab size if available
    if gpt_info.vocab_size:
        display += f", vocab:{gpt_info.vocab_size}"

    # Add date
    date_str = gpt_info.modified_date.strftime('%Y-%m-%d')
    display += f", {date_str}"

    display += ")"

    return display


def format_tokenizer_display(tokenizer_path: str) -> str:
    """
    Create a display-friendly string for tokenizer selection dropdown.

    Args:
        tokenizer_path: Path to tokenizer

    Returns:
        Formatted string for display
    """
    tok_info = get_tokenizer_info(tokenizer_path)
    if not tok_info:
        return Path(tokenizer_path).name

    display = tok_info.filename

    # Add vocab size and languages
    if tok_info.vocab_size:
        display += f" (vocab:{tok_info.vocab_size}"
        if tok_info.languages:
            display += f", {','.join(tok_info.languages[:2])}"  # First 2 languages
        display += ")"

    return display
