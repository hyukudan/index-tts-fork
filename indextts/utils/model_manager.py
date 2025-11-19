"""
Model Manager for IndexTTS - Hot-swap system for loading/unloading models efficiently.

This module provides a centralized system for managing TTS model instances,
handling GPU memory allocation, and enabling efficient model switching.
"""

import gc
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
from omegaconf import OmegaConf

from indextts.infer_v2 import IndexTTS2


logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Extended metadata for a TTS model."""
    path: str
    filename: str
    size_mb: float
    version: str
    languages: List[str]
    vocab_size: int
    model_dim: int
    num_layers: int
    num_heads: int
    tokenizer_path: Optional[str]
    config_path: Optional[str]
    recommended_vram_gb: float
    architecture: str
    last_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        return cls(**data)


class ModelManager:
    """
    Manages IndexTTS2 model instances with hot-swap capability.

    Features:
    - Single active model in GPU memory
    - Efficient load/unload with VRAM cleanup
    - Model metadata extraction and caching
    - GPU memory estimation
    - Tokenizer auto-detection
    """

    def __init__(self, registry_path: Optional[str] = None, auto_unload_timeout: Optional[int] = None):
        """
        Initialize the Model Manager.

        Args:
            registry_path: Path to model registry JSON. Defaults to ~/.indextts/model_registry.json
            auto_unload_timeout: Optional timeout in seconds for auto-unload (None = disabled)
        """
        self.current_model: Optional[IndexTTS2] = None
        self.current_model_path: Optional[str] = None
        self.current_metadata: Optional[ModelMetadata] = None
        self.last_activity_time: Optional[float] = None
        self.auto_unload_timeout = auto_unload_timeout

        # Registry for model metadata
        if registry_path is None:
            registry_dir = Path.home() / ".indextts"
            registry_dir.mkdir(exist_ok=True)
            registry_path = str(registry_dir / "model_registry.json")

        self.registry_path = registry_path
        self.registry: Dict[str, ModelMetadata] = self._load_registry()

        logger.info(f"ModelManager initialized. Registry: {self.registry_path}")
        if auto_unload_timeout:
            logger.info(f"Auto-unload enabled: {auto_unload_timeout}s timeout")

    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load model registry from disk."""
        if not os.path.exists(self.registry_path):
            return {}

        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            return {k: ModelMetadata.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}. Starting fresh.")
            return {}

    def _save_registry(self):
        """Save model registry to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.registry.items()}
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def extract_model_metadata(
        self,
        gpt_path: str,
        config_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ) -> ModelMetadata:
        """
        Extract metadata from a model checkpoint.

        Args:
            gpt_path: Path to GPT checkpoint (.pth file)
            config_path: Path to config.yaml (auto-detected if None)
            tokenizer_path: Path to tokenizer (auto-detected if None)

        Returns:
            ModelMetadata object with extracted information
        """
        gpt_path = str(Path(gpt_path).resolve())

        # Check cache first
        if gpt_path in self.registry:
            logger.info(f"Using cached metadata for {Path(gpt_path).name}")
            return self.registry[gpt_path]

        logger.info(f"Extracting metadata from {gpt_path}")

        # Basic file info
        file_size = os.path.getsize(gpt_path) / (1024 ** 2)  # MB
        filename = os.path.basename(gpt_path)

        # Auto-detect config
        if config_path is None:
            model_dir = os.path.dirname(gpt_path)
            potential_config = os.path.join(model_dir, "config.yaml")
            if os.path.exists(potential_config):
                config_path = potential_config

        # Load checkpoint to extract architecture details
        try:
            checkpoint = torch.load(gpt_path, map_location='cpu', weights_only=False)

            # Extract vocab size
            if 'text_embedding.weight' in checkpoint:
                vocab_size = checkpoint['text_embedding.weight'].shape[0]
            else:
                vocab_size = 12000  # Default for v2

            # Extract model dimensions
            if 'text_pos_embedding.emb.weight' in checkpoint:
                model_dim = checkpoint['text_pos_embedding.emb.weight'].shape[1]
            else:
                model_dim = 1280  # Default

            # Count layers
            layer_keys = [k for k in checkpoint.keys() if 'gpt.h.' in k]
            num_layers = len(set(k.split('.')[2] for k in layer_keys)) if layer_keys else 24

            # Extract num_heads (from attention weights)
            num_heads = 20  # Default
            for key in checkpoint.keys():
                if 'attn.c_attn.weight' in key:
                    attn_weight = checkpoint[key]
                    # c_attn has 3 * hidden_dim (Q, K, V concatenated)
                    if attn_weight.shape[0] == 3 * model_dim:
                        num_heads = model_dim // 64  # Assuming head_dim=64
                    break

            del checkpoint

        except Exception as e:
            logger.warning(f"Could not load checkpoint for metadata: {e}. Using defaults.")
            vocab_size = 12000
            model_dim = 1280
            num_layers = 24
            num_heads = 20

        # Load config if available
        version = "2.0"
        languages = ["zh", "en"]  # Default

        if config_path and os.path.exists(config_path):
            try:
                cfg = OmegaConf.load(config_path)
                version = str(cfg.get('version', '2.0'))

                # Try to detect languages from config or filename
                if 'languages' in cfg:
                    languages = cfg.languages
                elif 'chinese' in filename.lower():
                    languages = ["zh"]
                elif 'english' in filename.lower():
                    languages = ["en"]

            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        # Auto-detect tokenizer
        if tokenizer_path is None:
            model_dir = os.path.dirname(gpt_path)

            # Check same directory
            for bpe_file in ['bpe.model', 'tokenizer.model', f'{Path(gpt_path).stem}_bpe.model']:
                candidate = os.path.join(model_dir, bpe_file)
                if os.path.exists(candidate):
                    tokenizer_path = candidate
                    break

            # Check tokenizers subdirectory
            if tokenizer_path is None:
                tokenizers_dir = os.path.join(model_dir, 'tokenizers')
                if os.path.exists(tokenizers_dir):
                    bpe_files = list(Path(tokenizers_dir).glob('*.model'))
                    if bpe_files:
                        tokenizer_path = str(bpe_files[0])

        # Estimate VRAM requirements
        # Base model size + embeddings + working memory
        recommended_vram_gb = (file_size / 1024) * 2.5  # 2.5x model size for safety

        metadata = ModelMetadata(
            path=gpt_path,
            filename=filename,
            size_mb=file_size,
            version=version,
            languages=languages,
            vocab_size=vocab_size,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            tokenizer_path=tokenizer_path,
            config_path=config_path,
            recommended_vram_gb=recommended_vram_gb,
            architecture=f"UnifiedVoice-v{version}",
            last_used=None
        )

        # Cache metadata
        self.registry[gpt_path] = metadata
        self._save_registry()

        return metadata

    def unload_current_model(self) -> bool:
        """
        Unload the currently loaded model and free GPU memory.

        Returns:
            True if a model was unloaded, False if no model was loaded
        """
        if self.current_model is None:
            logger.info("No model currently loaded")
            return False

        logger.info(f"Unloading model: {self.current_metadata.filename}")

        # Delete model instance
        del self.current_model
        self.current_model = None
        self.current_model_path = None
        self.current_metadata = None
        self.last_activity_time = None

        # Aggressive garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Model unloaded and GPU memory freed")
        return True

    def load_model(
        self,
        gpt_path: str,
        gpu_id: int = 0,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        config_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs
    ) -> IndexTTS2:
        """
        Load a model with hot-swap (unloads current model first).

        Args:
            gpt_path: Path to GPT checkpoint
            gpu_id: GPU device ID (0 for first GPU, 1 for second, etc.)
            use_fp16: Use FP16 precision
            use_cuda_kernel: Use BigVGAN CUDA kernel
            config_path: Optional config path (auto-detected if None)
            tokenizer_path: Optional tokenizer path (auto-detected if None)
            **kwargs: Additional arguments for IndexTTS2

        Returns:
            Loaded IndexTTS2 instance
        """
        gpt_path = str(Path(gpt_path).resolve())

        # Check if this model is already loaded
        if self.current_model_path == gpt_path:
            logger.info(f"Model {Path(gpt_path).name} already loaded")
            return self.current_model

        # Unload current model first
        if self.current_model is not None:
            logger.info(f"Switching from {self.current_metadata.filename} to {Path(gpt_path).name}")
            self.unload_current_model()

        # Extract/load metadata
        metadata = self.extract_model_metadata(gpt_path, config_path, tokenizer_path)

        # Use metadata paths if not explicitly provided
        if config_path is None:
            config_path = metadata.config_path
        if tokenizer_path is None:
            tokenizer_path = metadata.tokenizer_path

        # Determine model directory
        model_dir = os.path.dirname(gpt_path)

        # Construct device string
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading {metadata.filename} on {device}")
        logger.info(f"  Architecture: {metadata.architecture}")
        logger.info(f"  Languages: {', '.join(metadata.languages)}")
        logger.info(f"  Tokenizer: {Path(tokenizer_path).name if tokenizer_path else 'auto'}")
        logger.info(f"  Estimated VRAM: {metadata.recommended_vram_gb:.1f} GB")

        start_time = time.time()

        # Load model
        try:
            model = IndexTTS2(
                cfg_path=config_path or os.path.join(model_dir, "config.yaml"),
                model_dir=model_dir,
                use_fp16=use_fp16,
                device=device,
                use_cuda_kernel=use_cuda_kernel,
                gpt_checkpoint_path=gpt_path,
                bpe_model_path=tokenizer_path,
                **kwargs
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")

            # Update metadata
            metadata.last_used = datetime.now().isoformat()

            # Store current model
            self.current_model = model
            self.current_model_path = gpt_path
            self.current_metadata = metadata
            self.last_activity_time = time.time()

            # Save updated registry
            self._save_registry()

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_current_model(self) -> Optional[IndexTTS2]:
        """Get the currently loaded model instance."""
        return self.current_model

    def get_current_metadata(self) -> Optional[ModelMetadata]:
        """Get metadata for the currently loaded model."""
        return self.current_metadata

    def list_models(self) -> List[ModelMetadata]:
        """List all models in the registry."""
        return list(self.registry.values())

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with 'allocated_gb', 'reserved_gb', 'free_gb'
        """
        if not torch.cuda.is_available():
            return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0}

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        device_props = torch.cuda.get_device_properties(0)
        total = device_props.total_memory / (1024 ** 3)
        free = total - reserved

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'total_gb': total
        }

    def mark_activity(self):
        """Mark current time as last activity (e.g., after generation)."""
        if self.current_model is not None:
            self.last_activity_time = time.time()
            logger.debug(f"Activity marked for {self.current_metadata.filename}")

    def get_idle_time(self) -> Optional[float]:
        """
        Get seconds since last activity.

        Returns:
            Idle time in seconds, or None if no model is loaded
        """
        if self.last_activity_time is None:
            return None
        return time.time() - self.last_activity_time

    def check_auto_unload(self) -> bool:
        """
        Check if model should be auto-unloaded due to timeout.

        Returns:
            True if model was unloaded, False otherwise
        """
        if self.auto_unload_timeout is None or self.current_model is None:
            return False

        idle_time = self.get_idle_time()
        if idle_time and idle_time > self.auto_unload_timeout:
            logger.info(f"Auto-unloading model after {idle_time:.0f}s idle time")
            self.unload_current_model()
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the model manager.

        Returns:
            Dictionary with model status, memory usage, and activity info
        """
        memory = self.get_memory_usage()
        idle_time = self.get_idle_time()

        status = {
            'model_loaded': self.current_model is not None,
            'model_name': self.current_metadata.filename if self.current_metadata else None,
            'model_path': self.current_model_path,
            'languages': self.current_metadata.languages if self.current_metadata else [],
            'vram_allocated_gb': memory['allocated_gb'],
            'vram_reserved_gb': memory['reserved_gb'],
            'vram_free_gb': memory['free_gb'],
            'vram_total_gb': memory.get('total_gb', 0),
            'vram_usage_pct': (memory['reserved_gb'] / memory.get('total_gb', 1)) * 100 if memory.get('total_gb', 0) > 0 else 0,
            'idle_time_seconds': idle_time,
            'idle_time_formatted': self._format_time(idle_time) if idle_time else None,
            'auto_unload_enabled': self.auto_unload_timeout is not None,
            'auto_unload_timeout': self.auto_unload_timeout,
        }

        return status

    def get_vram_warning(self) -> Optional[str]:
        """
        Get VRAM usage warning if usage is high.

        Returns:
            Warning message if VRAM > 90%, None otherwise
        """
        status = self.get_status()
        usage_pct = status['vram_usage_pct']

        if usage_pct > 95:
            return f"⚠️ CRITICAL: VRAM usage at {usage_pct:.1f}%. Consider unloading the model."
        elif usage_pct > 90:
            return f"⚠️ WARNING: VRAM usage at {usage_pct:.1f}%. High memory pressure."
        elif usage_pct > 80:
            return f"ℹ️ INFO: VRAM usage at {usage_pct:.1f}%. Memory usage is elevated."

        return None

    @staticmethod
    def _format_time(seconds: Optional[float]) -> str:
        """Format seconds into human-readable time."""
        if seconds is None:
            return "N/A"

        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
