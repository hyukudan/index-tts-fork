#!/usr/bin/env python3
"""
Preprocess Catalan Arrow datasets (lafrescat/festcat) for IndexTTS2 fine-tuning.

This script reads HuggingFace Arrow datasets and performs the complete IndexTTS preprocessing:
  1. Extract audio from Arrow format
  2. Text normalization and tokenization (Catalan)
  3. Semantic feature extraction via SeamlessM4T + Wav2Vec2Bert
  4. Semantic code quantization with MaskGCT semantic codec
  5. Conditioning latent + emotion vector extraction with UnifiedVoice v2
  6. Generate train/validation JSONL manifests

Usage:
    python tools/preprocess_catalan.py \\
        --dataset ca=/home/sergioc/projects/datasets/catalan/lafrescat \\
        --dataset ca=/home/sergioc/projects/datasets/catalan/festcat \\
        --output-root processed_catalan \\
        --tokenizer checkpoints/bpe.model \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model
from huggingface_hub import hf_hub_download
import safetensors.torch

try:
    from datasets import load_from_disk, Audio
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def load_existing_ids(manifest_path: Path) -> set[str]:
    ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ids.add(record["id"])
    return ids


def update_stats_file(
    stats_path: Path,
    train_ids: set[str],
    val_ids: set[str],
    tokenizer_path: Path,
    checkpoint_path: Path,
) -> None:
    stats = {
        "total": len(train_ids) + len(val_ids),
        "train": len(train_ids),
        "val": len(val_ids),
        "tokenizer": str(tokenizer_path),
        "gpt_checkpoint": str(checkpoint_path),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as stats_f:
        json.dump(stats, stats_f, indent=2, ensure_ascii=False)


def assign_to_validation(sample_id: str, ratio: float) -> bool:
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    value = int(digest, 16) % 1_000_000
    return (value / 1_000_000) < ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Catalan Arrow datasets for IndexTTS2 fine-tuning."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="LANG=DATASET_PATH[=OUTPUT]",
        help=(
            "Arrow dataset to process. Provide entries like "
            "`ca=/path/to/lafrescat` or "
            "`ca=/path/to/lafrescat=lafrescat_processed`. "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_catalan"),
        help="Base directory for outputs.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/bpe.model"),
        help="Path to the trained SentencePiece model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/config.yaml"),
        help="IndexTTS config YAML (used to instantiate UnifiedVoice).",
    )
    parser.add_argument(
        "--gpt-checkpoint",
        type=Path,
        default=Path("checkpoints/gpt.pth"),
        help="Base UnifiedVoice checkpoint for conditioning extraction.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device (cuda or cpu).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of data reserved for validation (default: 5%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for split shuffling.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples for debugging (0 means process all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose feature files already exist.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples to process concurrently.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of background worker threads for audio loading.",
    )
    return parser.parse_args()


SPEAKER_PATTERN = re.compile(r"^\s*(?:speaker|spk)\s*\d+\s*[:：]\s*", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = SPEAKER_PATTERN.sub("", text)
    return text.strip()


def load_audio_from_array(audio_array: np.ndarray, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load audio from numpy array (from Arrow dataset)."""
    # Convert to torch tensor
    wav = torch.from_numpy(audio_array).float()

    # Ensure it's 2D (channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # Convert to mono if needed
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr


class SemanticExtractor:
    def __init__(self, stats_path: Path, device: torch.device):
        self.device = device
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            path_=stats_path
        )
        self.semantic_model = self.semantic_model.to(device)
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        self.semantic_model.eval()

    @torch.inference_mode()
    def extract(
        self,
        waveforms: Sequence[torch.Tensor] | torch.Tensor,
        sample_rates: Sequence[int] | int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(waveforms, torch.Tensor):
            waveforms = [waveforms]
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates]

        arrays: List[np.ndarray] = []
        for wav, sr in zip(waveforms, sample_rates):
            current = wav
            if sr != 16000:
                current = torchaudio.functional.resample(current, sr, 16000)
            arrays.append(current.squeeze(0).cpu().numpy())

        inputs = self.feature_extractor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = outputs.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat, attention_mask


def build_unified_voice(cfg, checkpoint: Path, device: torch.device) -> UnifiedVoice:
    gpt = UnifiedVoice(**cfg.gpt)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    gpt.load_state_dict(state, strict=False)
    gpt = gpt.to(device)
    gpt.eval()
    return gpt


def ensure_dirs(root: Path) -> Dict[str, Path]:
    subdirs = {
        "codes": root / "codes",
        "condition": root / "condition",
        "emo": root / "emo_vec",
        "text": root / "text_ids",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def process_batch(
    samples: Sequence[Dict[str, Any]],
    tokenizer: TextTokenizer,
    semantic_codec,
    semantic_extractor: SemanticExtractor,
    gpt: UnifiedVoice,
    dirs: Dict[str, Path],
    executor: ThreadPoolExecutor | None,
) -> Tuple[List[Dict[str, Any]], int]:
    prepared: List[Dict[str, Any]] = []
    skipped = 0

    candidates: List[Dict[str, Any]] = []
    for sample in samples:
        # Extract text from 'transcription' field (Arrow format)
        text = clean_text(sample.get("transcription", sample.get("text", "")))

        # Tokenize for Catalan (using ca language hint)
        text_tokens = tokenizer.tokenize(text, language="ca")
        if not text_tokens:
            skipped += 1
            continue
        text_ids = np.asarray(tokenizer.convert_tokens_to_ids(text_tokens), dtype=np.int32)

        # Audio is embedded in Arrow format
        audio_data = sample.get("audio")
        if audio_data is None:
            skipped += 1
            continue

        candidates.append(
            {
                "sample": sample,
                "text": text,
                "text_ids": text_ids,
                "audio_data": audio_data,
            }
        )

    if not candidates:
        return [], skipped

    # Load audio from Arrow format
    for item in candidates:
        try:
            audio_data = item["audio_data"]
            # Arrow audio format: dict with 'array' and 'sampling_rate'
            audio_array = audio_data["array"]
            sr = audio_data["sampling_rate"]

            waveform, sr = load_audio_from_array(audio_array, sr, target_sr=24000)
            item["waveform"] = waveform
            item["sr"] = sr
            prepared.append(item)
        except Exception as e:
            print(f"Error loading audio for sample {item['sample'].get('id', 'unknown')}: {e}")
            skipped += 1
            continue

    if not prepared:
        return [], skipped

    waveforms = [item["waveform"] for item in prepared]
    sample_rates = [item["sr"] for item in prepared]
    feat, attention_mask = semantic_extractor.extract(waveforms, sample_rates)

    with torch.inference_mode():
        semantic_code, _ = semantic_codec.quantize(feat)
        if semantic_code.dim() == 1:
            semantic_code = semantic_code.unsqueeze(0)
        semantic_code = semantic_code.detach().cpu().numpy().astype(np.int32)
        cond_lengths = attention_mask.sum(dim=1).long()
        feat_t = feat.transpose(1, 2)
        cond_lengths_device = cond_lengths.to(feat.device)
        conditioning = gpt.get_conditioning(feat_t, cond_lengths_device)
        emo_vec = gpt.get_emovec(feat, cond_lengths_device)

    conditioning_np = conditioning.detach().cpu().numpy().astype(np.float32)
    emo_vec_np = emo_vec.detach().cpu().numpy().astype(np.float32)

    entries: List[Dict[str, Any]] = []
    output_root = dirs["codes"].parent
    for idx, item in enumerate(prepared):
        sample = item["sample"]
        uid = sample["id"]
        code_path = dirs["codes"] / f"{uid}.npy"
        cond_path = dirs["condition"] / f"{uid}.npy"
        emo_path = dirs["emo"] / f"{uid}.npy"
        text_path = dirs["text"] / f"{uid}.npy"

        save_numpy(code_path, semantic_code[idx])
        save_numpy(cond_path, conditioning_np[idx])
        save_numpy(emo_path, emo_vec_np[idx])
        save_numpy(text_path, item["text_ids"])

        entry = {
            "id": uid,
            "audio_path": f"arrow://{sample.get('id', 'unknown')}",  # Placeholder since audio is embedded
            "text": item["text"],
            "speaker": sample.get("speaker_id", "unknown"),
            "language": "ca",
            "duration": len(item["waveform"][0]) / item["sr"],
            "text_ids_path": text_path.relative_to(output_root).as_posix(),
            "text_len": int(item["text_ids"].size),
            "codes_path": code_path.relative_to(output_root).as_posix(),
            "code_len": int(semantic_code[idx].size),
            "condition_path": cond_path.relative_to(output_root).as_posix(),
            "condition_len": int(conditioning_np[idx].shape[0]),
            "emo_vec_path": emo_path.relative_to(output_root).as_posix(),
        }
        entries.append(entry)

    return entries, skipped


def parse_dataset_spec(spec: str, output_root: Path) -> tuple[str, Path, Path]:
    parts = spec.split("=")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            f"Invalid --dataset entry '{spec}'. Expected format LANG=DATASET_PATH or LANG=DATASET_PATH=OUTPUT."
        )
    lang = parts[0].strip()
    dataset_path = Path(parts[1].strip())
    if len(parts) == 3 and parts[2].strip():
        output_dir = Path(parts[2].strip())
    else:
        # Use dataset folder name for output
        output_dir = output_root / dataset_path.name
    return lang, dataset_path, output_dir


def preprocess_dataset(
    dataset_path: Path,
    output_dir: Path,
    dataset_language: str,
    normalizer_hint: Optional[str],
    tokenizer_path: Path,
    cfg,
    device: torch.device,
    semantic_extractor: SemanticExtractor,
    semantic_codec,
    gpt: UnifiedVoice,
    args,
    batch_size: int,
    executor: Optional[ThreadPoolExecutor],
) -> tuple[int, int, int, int]:
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library not found. Install with: uv sync --extra datasets")

    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading Arrow dataset from {dataset_path}...")
    # Load HuggingFace Arrow dataset
    # The dataset folder should contain dataset/data-00000-of-00001.arrow
    dataset_folder = dataset_path / "dataset"
    if not dataset_folder.exists():
        dataset_folder = dataset_path

    dataset = load_from_disk(str(dataset_folder))
    print(f"Loaded {len(dataset)} samples")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(output_dir)

    tokenizer = TextTokenizer(
        str(tokenizer_path),
        TextNormalizer(preferred_language=normalizer_hint),
    )

    train_manifest_path = output_dir / "train_manifest.jsonl"
    val_manifest_path = output_dir / "val_manifest.jsonl"
    stats_output_path = output_dir / "stats.json"

    train_ids = load_existing_ids(train_manifest_path)
    val_ids = load_existing_ids(val_manifest_path)

    train_file = open(train_manifest_path, "a", encoding="utf-8")
    val_file = open(val_manifest_path, "a", encoding="utf-8")

    processed = 0
    skipped = 0
    pending: List[Dict[str, Any]] = []

    def flush(force: bool = False) -> None:
        nonlocal pending, processed, skipped
        while pending and (
            force
            or len(pending) >= batch_size
            or (args.max_samples and processed + len(pending) >= args.max_samples)
        ):
            limit = batch_size
            if args.max_samples:
                remaining = args.max_samples - processed
                if remaining <= 0:
                    pending.clear()
                    return
                limit = min(limit, remaining)
            batch_records = pending[:limit]
            entries, batch_skipped = process_batch(
                batch_records,
                tokenizer,
                semantic_codec,
                semantic_extractor,
                gpt,
                dirs,
                executor=executor,
            )
            skipped += batch_skipped
            pending = pending[limit:]
            for entry in entries:
                is_val = assign_to_validation(entry["id"], args.val_ratio)
                if is_val:
                    if entry["id"] not in val_ids:
                        val_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        val_file.flush()
                        val_ids.add(entry["id"])
                else:
                    if entry["id"] not in train_ids:
                        train_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        train_file.flush()
                        train_ids.add(entry["id"])
                processed += 1
                if args.max_samples and processed >= args.max_samples:
                    pending.clear()
                    return

    try:
        # Process Arrow dataset samples
        for idx, sample in enumerate(tqdm(dataset, desc=f"Preprocessing [{dataset_language}]", unit="sample")):
            if args.max_samples and processed >= args.max_samples:
                break

            # Generate unique ID if not present
            if "id" not in sample:
                sample["id"] = f"{dataset_path.name}_{idx:06d}"

            uid = sample["id"]

            # Skip if already processed
            if uid in train_ids or uid in val_ids:
                skipped += 1
                continue

            if args.skip_existing and (
                (output_dir / "codes" / f"{uid}.npy").exists()
                and (output_dir / "text_ids" / f"{uid}.npy").exists()
            ):
                skipped += 1
                continue

            pending.append(dict(sample))
            flush()
            if args.max_samples and processed >= args.max_samples:
                break
        flush(force=True)
    finally:
        train_file.close()
        val_file.close()

    update_stats_file(
        stats_output_path,
        train_ids,
        val_ids,
        tokenizer_path,
        args.gpt_checkpoint,
    )

    print(
        f"[{dataset_language}] processed={processed} skipped={skipped} "
        f"train={len(train_ids)} val={len(val_ids)} -> {output_dir}"
    )

    return processed, skipped, len(train_ids), len(val_ids)


def main() -> None:
    args = parse_args()

    if not args.dataset:
        raise ValueError(
            "No datasets specified. Use --dataset ca=/path/to/dataset. "
            "Example: --dataset ca=/home/sergioc/projects/datasets/catalan/lafrescat"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch_size = max(1, args.batch_size)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_specs: List[tuple[str, Path, Path]] = []
    for spec in args.dataset:
        lang, dataset_path, output_dir = parse_dataset_spec(spec, output_root)
        dataset_specs.append((lang, dataset_path, output_dir))

    executor: Optional[ThreadPoolExecutor] = None
    if args.workers > 0:
        executor = ThreadPoolExecutor(max_workers=args.workers)

    cfg = OmegaConf.load(args.config)
    stats_value = OmegaConf.select(cfg, "w2v_stat")
    stats_path = Path(stats_value or "checkpoints/wav2vec2bert_stats.pt")
    if not stats_path.is_absolute():
        stats_path = (args.config.parent / stats_path).resolve()

    print("Initializing semantic extractor...")
    semantic_extractor = SemanticExtractor(stats_path, device)

    print("Loading semantic codec...")
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
    )
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device)
    semantic_codec.eval()

    print("Loading UnifiedVoice GPT...")
    gpt = build_unified_voice(cfg, args.gpt_checkpoint, device)

    summaries: List[tuple[str, int, int, int, int]] = []
    try:
        for lang, dataset_path, output_dir in dataset_specs:
            # For Catalan, use 'ca' language hint (will fall back to 'en' in normalizer)
            normalizer_hint = "ca" if lang.lower() == "ca" else lang.lower()
            processed, skipped, train_count, val_count = preprocess_dataset(
                dataset_path,
                output_dir,
                lang,
                normalizer_hint,
                args.tokenizer,
                cfg,
                device,
                semantic_extractor,
                semantic_codec,
                gpt,
                args,
                batch_size,
                executor,
            )
            summaries.append((dataset_path.name, processed, skipped, train_count, val_count))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if len(summaries) > 0:
        print("\n=== Summary ===")
        for name, processed, skipped, train_count, val_count in summaries:
            print(
                f"[{name}] processed={processed} skipped={skipped} "
                f"train={train_count} val={val_count}"
            )


if __name__ == "__main__":
    main()
