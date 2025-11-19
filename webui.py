import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to use (for multi-GPU systems)")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

# Configure environment paths before importing heavy dependencies so child modules see them
hf_cache_dir = os.path.join(cmd_args.model_dir, "hf_cache")
torch_cache_dir = os.path.join(cmd_args.model_dir, "torch_cache")
os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")
os.environ.setdefault("HF_HOME", hf_cache_dir)
os.environ.setdefault("HF_HUB_CACHE", hf_cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
os.environ.setdefault("TORCH_HOME", torch_cache_dir)
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(torch_cache_dir, exist_ok=True)

import gradio as gr
from indextts import infer
from indextts.infer_v2_modded import IndexTTS2
from tools.i18n.i18n import I18nAuto
from modelscope.hub import api

i18n = I18nAuto(language="Auto")
MODE = 'local'

# Import GPU configuration system and utilities
from indextts.utils.gpu_config import setup_gpu, GPUConfig
from indextts.utils.resource_monitor import get_monitor, format_vram_bar
from indextts.utils.model_metadata import get_gpt_info, get_tokenizer_info
from indextts.utils.audio_history import get_history_manager
from indextts.utils.model_manager import ModelManager, ModelMetadata
from indextts.utils.model_comparison import ModelComparator
from indextts.utils.training_monitor import (
    TensorBoardManager,
    TrainingLogParser,
    TrainingAnalyzer,
    ExperimentTracker,
    find_training_logs,
    get_tensorboard_logdir,
    export_plot_to_png
)
from indextts.utils.emotion_presets import (
    get_preset_choices,
    get_preset_vector,
    mix_emotions,
    get_preset_description
)
from indextts.utils.duration_estimator import (
    estimate_duration,
    get_duration_display,
    format_duration
)
import gc

def cleanup_gpu_memory():
    """Clean up GPU memory to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()

# Setup GPU with interactive selection if needed
gpu_id, gpu_info = setup_gpu(cmd_args.gpu)

# Apply optimizations based on GPU architecture
if gpu_info.get('is_blackwell'):
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

# Dynamic TTS instance management
_PRIMARY_TTS = None
_MODEL_SELECTION = {"gpt": None, "bpe": None}
_gpu_config_manager = GPUConfig()
_current_gpu_id = gpu_id

# Initialize ModelManager for hot-swap functionality
_model_manager = ModelManager()
_model_comparator = ModelComparator(_model_manager)

# Initialize TensorBoard manager for training monitoring
_tensorboard_manager = TensorBoardManager()
_training_analyzer = TrainingAnalyzer()
_experiment_tracker = ExperimentTracker()

logger = logging.getLogger(__name__)


def _discover_gpt_checkpoints():
    """Discover available GPT checkpoints."""
    bases = [Path(cmd_args.model_dir), Path(current_dir) / "models"]
    results = []
    for base in bases:
        if not base.exists():
            continue
        for path in base.glob("*.pth"):
            name = path.name.lower()
            # Exclude non-GPT models
            if any(x in name for x in ("s2mel", "campplus", "bigvgan", "wav2vec", "emo", "spk", "cfm")):
                continue
            results.append(str(path.resolve()))
    results.sort()
    return results


def _discover_bpe_models():
    """Discover available BPE tokenizer models."""
    bases = [Path(cmd_args.model_dir), Path(current_dir) / "tokenizers"]
    results = []
    for base in bases:
        if not base.exists():
            continue
        for path in base.glob("*.model"):
            results.append(str(path.resolve()))
    results.sort()
    return results


def dispose_primary_tts():
    """Dispose the current TTS instance and free memory."""
    global _PRIMARY_TTS
    if _PRIMARY_TTS is not None:
        try:
            if hasattr(_PRIMARY_TTS, "gr_progress"):
                _PRIMARY_TTS.gr_progress = None
        finally:
            _PRIMARY_TTS = None

    # Use ModelManager to unload current model
    _model_manager.unload_current_model()


def load_primary_tts(gpt_path, bpe_path=None):
    """Load TTS with specified model paths using ModelManager."""
    global _PRIMARY_TTS

    resolved_gpt = os.path.abspath(gpt_path)
    resolved_bpe = os.path.abspath(bpe_path) if bpe_path else None
    previous_selection = _MODEL_SELECTION.copy()
    _MODEL_SELECTION["gpt"] = resolved_gpt
    _MODEL_SELECTION["bpe"] = resolved_bpe

    try:
        # Use ModelManager for hot-swap loading
        # If bpe_path is None, ModelManager will auto-detect tokenizer
        tts = _model_manager.load_model(
            gpt_path=resolved_gpt,
            gpu_id=_current_gpu_id,
            use_fp16=cmd_args.is_fp16,
            use_cuda_kernel=False,
            config_path=os.path.join(cmd_args.model_dir, "config.yaml"),
            tokenizer_path=resolved_bpe  # None triggers auto-detection
        )

        # Update bpe path in selection with auto-detected tokenizer
        if bpe_path is None:
            metadata = _model_manager.get_current_metadata()
            if metadata and metadata.tokenizer_path:
                _MODEL_SELECTION["bpe"] = metadata.tokenizer_path
    except Exception:
        _MODEL_SELECTION.update(previous_selection)
        raise

    # Keep reference for compatibility with existing code
    _PRIMARY_TTS = tts
    return tts


def ensure_primary_tts():
    """Ensure TTS is loaded, raise error if not."""
    if _PRIMARY_TTS is None:
        raise RuntimeError("No model loaded. Use the Load button in the UI.")
    return _PRIMARY_TTS


def get_tts():
    """Get current TTS instance or load default using ModelManager."""
    global _PRIMARY_TTS

    # First check if ModelManager has a model loaded
    current_model = _model_manager.get_current_model()
    if current_model is not None:
        _PRIMARY_TTS = current_model
        return current_model

    # Load default models on first access
    if _PRIMARY_TTS is None:
        default_gpt = os.path.join(cmd_args.model_dir, "gpt.pth")
        default_bpe = os.path.join(cmd_args.model_dir, "bpe.model")
        if os.path.exists(default_gpt) and os.path.exists(default_bpe):
            _PRIMARY_TTS = load_primary_tts(default_gpt, default_bpe)

    return _PRIMARY_TTS


# Initialize with default models
tts = get_tts()


def get_available_gpus():
    """Get list of available GPUs."""
    gpus = _gpu_config_manager.detect_gpus()
    return [(f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)", gpu['id']) for gpu in gpus]


def get_gpu_monitor_text():
    """Get formatted GPU monitoring information."""
    monitor = get_monitor()
    stats = monitor.get_gpu_stats(_current_gpu_id)

    if not stats:
        return "⚠️ GPU monitoring unavailable"

    lines = [
        f"**GPU {stats.device_id}: {stats.name}**",
        f"VRAM: {stats.memory_used}MB / {stats.memory_total}MB",
        format_vram_bar(stats.memory_percent, width=30),
    ]

    if stats.temperature is not None:
        temp_emoji = "🌡️" if stats.temperature < 80 else "🔥"
        lines.append(f"{temp_emoji} Temperature: {stats.temperature}°C")

    if stats.utilization is not None:
        lines.append(f"⚡ Utilization: {stats.utilization}%")

    risk = monitor.predict_oom_risk(_current_gpu_id)
    if risk in ("high", "critical"):
        lines.append(f"⚠️ OOM Risk: {risk.upper()}")

    return "\n".join(lines)


def get_model_status_text():
    """Get formatted model status with persistence info."""
    status = _model_manager.get_status()

    if not status['model_loaded']:
        return "⚪ **Model Status:** No model loaded"

    lines = [f"🟢 **Model Loaded:** {status['model_name']}"]

    # Languages
    if status['languages']:
        lines.append(f"🌍 Languages: {', '.join(status['languages'])}")

    # Idle time
    if status['idle_time_formatted']:
        idle_emoji = "⏱️" if status.get('idle_time_seconds', 0) < 300 else "💤"
        lines.append(f"{idle_emoji} Idle: {status['idle_time_formatted']}")

    # Auto-unload status
    if status['auto_unload_enabled']:
        lines.append(f"⏰ Auto-unload: {status['auto_unload_timeout']}s")

    # VRAM warning if applicable
    warning = _model_manager.get_vram_warning()
    if warning:
        lines.append(warning)

    return "\n".join(lines)


def unload_model_handler():
    """Handler for manual model unload button."""
    success = _model_manager.unload_current_model()
    if success:
        return "✅ Model unloaded successfully", get_model_status_text(), get_gpu_monitor_text()
    else:
        return "⚠️ No model was loaded", get_model_status_text(), get_gpu_monitor_text()


def refresh_status_handler():
    """Refresh both model and GPU status."""
    # Check auto-unload
    _model_manager.check_auto_unload()
    return get_model_status_text(), get_gpu_monitor_text()


def get_model_choices_with_metadata():
    """Get model choices with metadata for dropdown."""
    gpt_checkpoints = _discover_gpt_checkpoints()
    choices = []

    for gpt_path in gpt_checkpoints:
        try:
            metadata = _model_manager.extract_model_metadata(gpt_path)
            label = f"{metadata.filename} ({metadata.size_mb/1024:.1f}GB, v{metadata.version}, {'/'.join(metadata.languages)})"
            choices.append((label, gpt_path))
        except:
            choices.append((Path(gpt_path).name, gpt_path))

    return choices


def get_model_info_display(gpt_path):
    """Get formatted model information for display."""
    if not gpt_path:
        return "No model selected", "Select a model to see VRAM estimate"

    try:
        metadata = _model_manager.extract_model_metadata(gpt_path)

        # Auto-detected tokenizer
        tokenizer_info = f"**Auto-detected:** {Path(metadata.tokenizer_path).name if metadata.tokenizer_path else 'None'}"
        if metadata.tokenizer_path:
            tokenizer_info += f" ({metadata.vocab_size} vocab)"

        # VRAM estimate with color coding
        vram_gb = metadata.recommended_vram_gb
        if vram_gb < 12:
            vram_emoji = "🟢"
        elif vram_gb < 20:
            vram_emoji = "🟡"
        else:
            vram_emoji = "🔴"

        vram_info = f"{vram_emoji} **Estimated VRAM:** {vram_gb:.1f} GB"

        return tokenizer_info, vram_info

    except Exception as e:
        logger.warning(f"Failed to get model info: {e}")
        return "Error loading metadata", "VRAM estimate unavailable"


def get_gpu_selector_choices():
    """Get GPU choices for selector."""
    gpus = _gpu_config_manager.detect_gpus()
    choices = []

    for gpu in gpus:
        label = f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)"
        choices.append((label, gpu['id']))

    return choices


# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES = [
    "Match prompt audio",
    "Use emotion reference audio",
    "Use emotion vector",
    "Use emotion text description",
]
os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None
        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0)]
                             )


def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_sentence=120,
               target_duration=0,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")

    # Ensure TTS is loaded
    try:
        tts = ensure_primary_tts()
    except RuntimeError as exc:
        gr.Warning(str(exc))
        return gr.update()

    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0
    if emo_control_method == 1:
        emo_weight = emo_weight
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec_sum = sum([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
        if vec_sum > 1.5:
            gr.Warning("Emotion vector sum cannot exceed 1.5. Adjust the sliders and retry.")
            return
    else:
        vec = None

    print(f"Emo control mode:{emo_control_method},vec:{vec}")
    try:
        # Convert target_duration to duration_seconds (0 means None/auto)
        duration_seconds = float(target_duration) if target_duration and target_duration > 0 else None

        output = tts.infer(spk_audio_prompt=prompt, text=text,
                           output_path=output_path,
                           emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                           emo_vector=vec,
                           use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                           duration_seconds=duration_seconds,
                           verbose=cmd_args.verbose,
                           max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                           **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            # Clear CUDA cache and retry with lower memory settings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gr.Warning(f"GPU out of memory. Try reducing max_mel_tokens or max_text_tokens_per_sentence. Error: {e}")
            return gr.update()
        else:
            gr.Warning(f"Generation error: {e}")
            return gr.update()
    except Exception as e:
        gr.Warning(f"Unexpected error during generation: {e}")
        return gr.update()

    # Save to history
    try:
        history = get_history_manager()
        gpt_name = Path(_MODEL_SELECTION.get("gpt", "unknown")).name if _MODEL_SELECTION.get("gpt") else "unknown"
        bpe_name = Path(_MODEL_SELECTION.get("bpe", "unknown")).name if _MODEL_SELECTION.get("bpe") else "unknown"

        history.add_generation(
            audio_path=output,
            text=text,
            model_gpt=gpt_name,
            model_tokenizer=bpe_name,
            prompt_audio=prompt,
            emotion_mode=EMO_CHOICES[emo_control_method] if emo_control_method < len(EMO_CHOICES) else f"Mode {emo_control_method}",
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 50),
            max_mel_tokens=kwargs.get("max_mel_tokens", 1500),
        )
    except Exception as e:
        logger.warning(f"Failed to save to history: {e}")

    # Mark model activity for persistence tracking
    _model_manager.mark_activity()

    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    batch_rows_state = gr.State([])
    next_batch_id_state = gr.State(1)
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    # Model and GPU Configuration
    with gr.Accordion("Model & GPU Configuration", open=True):
        # GPU selection
        with gr.Row():
            gpu_choices = get_gpu_selector_choices()
            gpu_dropdown = gr.Dropdown(
                choices=gpu_choices,
                value=gpu_choices[0][1] if gpu_choices else 0,
                label="GPU Device",
                interactive=True,
                scale=2,
                info="Select GPU for model inference"
            )

        # Model selection with metadata
        with gr.Row():
            gpt_checkpoints = _discover_gpt_checkpoints()
            model_choices_with_metadata = get_model_choices_with_metadata()

            gpt_dropdown = gr.Dropdown(
                choices=[choice[0] for choice in model_choices_with_metadata],
                value=model_choices_with_metadata[0][0] if model_choices_with_metadata else None,
                label="Model Checkpoint",
                interactive=True,
                scale=3,
                info="Select TTS model (tokenizer auto-detected)"
            )
            load_models_button = gr.Button("Load Model", variant="primary", scale=1)

        # Model metadata display
        # Get initial metadata for default model
        initial_tokenizer_info = "**Tokenizer:** Select a model to see info"
        initial_vram_info = "**VRAM:** Select a model to see estimate"
        if model_choices_with_metadata:
            default_model_path = model_choices_with_metadata[0][1]
            initial_tokenizer_info, initial_vram_info = get_model_info_display(default_model_path)

        with gr.Row():
            with gr.Column(scale=1):
                tokenizer_info_display = gr.Markdown(value=initial_tokenizer_info)
            with gr.Column(scale=1):
                vram_info_display = gr.Markdown(value=initial_vram_info)

        model_status = gr.Markdown(value="✅ Default models loaded" if tts else "⚠️ No models loaded")

        # Model persistence status and controls
        with gr.Row():
            with gr.Column(scale=2):
                model_status_display = gr.Markdown(value=get_model_status_text(), label="Model Status")
            with gr.Column(scale=2):
                gpu_monitor_display = gr.Markdown(value=get_gpu_monitor_text(), label="GPU Monitor")
            with gr.Column(scale=1):
                with gr.Row():
                    unload_model_button = gr.Button("🗑️ Unload Model", variant="secondary", size="sm")
                with gr.Row():
                    refresh_status_button = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")

        # State for model paths (mapping from display labels to actual paths)
        gpt_paths_state = gr.State({choice[0]: choice[1] for choice in model_choices_with_metadata})

    with gr.Accordion("Emotion Settings", open=True):
        with gr.Row():
            emo_control_method = gr.Radio(
                choices=EMO_CHOICES,
                type="index",
                value=EMO_CHOICES[0],
                label="Emotion Control Mode",
            )

    with gr.Group(visible=False) as emotion_reference_group:
        with gr.Row():
            emo_upload = gr.Audio(label="Emotion Reference Audio", type="filepath")
        with gr.Row():
            emo_weight = gr.Slider(label="Emotion Weight", minimum=0.0, maximum=1.6, value=0.8, step=0.01)

    with gr.Row():
        emo_random = gr.Checkbox(label="Random Emotion Sampling", value=False, visible=False)

    with gr.Group(visible=False) as emotion_vector_group:
        # Emotion Presets
        with gr.Row():
            with gr.Column(scale=3):
                emotion_preset = gr.Dropdown(
                    choices=get_preset_choices(),
                    value="neutral",
                    label="✨ Emotion Presets",
                    info="Quick selection of common emotions"
                )
            with gr.Column(scale=1):
                apply_preset_btn = gr.Button("Apply Preset", variant="primary", size="sm")

        with gr.Accordion("Advanced: Custom Emotion Mix", open=False):
            with gr.Row():
                with gr.Column():
                    mix_preset_a = gr.Dropdown(
                        choices=[choice for choice in get_preset_choices() if choice[1] != "custom"],
                        value="neutral",
                        label="Emotion A",
                        scale=2
                    )
                with gr.Column(scale=1):
                    mix_ratio = gr.Slider(
                        label="A ← → B",
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=5,
                        info="0% = A only, 100% = B only"
                    )
                with gr.Column():
                    mix_preset_b = gr.Dropdown(
                        choices=[choice for choice in get_preset_choices() if choice[1] != "custom"],
                        value="happy",
                        label="Emotion B",
                        scale=2
                    )
            with gr.Row():
                apply_mix_btn = gr.Button("Apply Mix", variant="secondary", size="sm")
                preset_description = gr.Markdown(value=get_preset_description("neutral"))

        # Manual Sliders
        with gr.Accordion("Expert: Manual Vector Control", open=False):
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="Happiness", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec4 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec6 = gr.Slider(label="Fear", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec7 = gr.Slider(label="Arousal", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.4, value=0.0, step=0.05)

    with gr.Group(visible=False) as emo_text_group:
        with gr.Row():
            emo_text = gr.Textbox(label="Emotion Description", placeholder="Describe the target emotion", value="", info="e.g., happy, angry, sad")

    with gr.Accordion("Advanced Generation Settings", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**GPT2 Sampling Settings** _Parameters affect diversity and speed._")
                with gr.Row():
                    do_sample = gr.Checkbox(label="do_sample", value=True, info="Enable sampling")
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                with gr.Row():
                    top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                    top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                    num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                with gr.Row():
                    repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                    length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info="Maximum generated mel tokens")
            with gr.Column(scale=2):
                gr.Markdown("**Sentence Settings** _Controls sentence splitting._")
                with gr.Row():
                    max_text_tokens_per_sentence = gr.Slider(
                        label="Max tokens per sentence",
                        value=120,
                        minimum=20,
                        maximum=tts.cfg.gpt.max_text_tokens,
                        step=2,
                        key="max_text_tokens_per_sentence",
                        info="Higher values mean longer sentences; adjust between 80-200",
                    )
                with gr.Accordion("Preview sentences", open=True) as sentences_settings:
                    sentences_preview = gr.Dataframe(
                        headers=["Index", "Sentence", "Token Count"],
                        key="sentences_preview",
                        wrap=True,
                    )
        advanced_params = [
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
        ]

    with gr.Tabs():
        with gr.Tab("Single Generation"):
            with gr.Row():
                prompt_audio = gr.Audio(label="Prompt Audio", key="prompt_audio",
                                        sources=["upload", "microphone"], type="filepath")
                with gr.Column():
                    input_text_single = gr.TextArea(label="Text", key="input_text_single", placeholder="Enter text to synthesize", info=f"Model version {tts.model_version or '1.0'}")

                    # Duration estimation and control
                    with gr.Row():
                        with gr.Column(scale=2):
                            duration_estimate_display = gr.Markdown(value="📊 Enter text to see duration estimate", label="Duration Preview")
                        with gr.Column(scale=1):
                            target_duration = gr.Number(
                                label="Target Duration (seconds)",
                                value=0,
                                minimum=0,
                                maximum=30,
                                step=0.1,
                                info="0 = auto"
                            )

                    gen_button = gr.Button("Generate", key="gen_button", interactive=True)

            output_audio = gr.Audio(label="Generated Result", visible=True, key="output_audio")

            # Waveform visualization
            with gr.Row():
                waveform_plot = gr.Plot(label="Audio Waveform", visible=False)

            if len(example_cases) > 0:
                gr.Examples(
                    examples=example_cases,
                    examples_per_page=20,
                    inputs=[prompt_audio,
                            emo_control_method,
                            input_text_single,
                            emo_upload,
                            emo_weight,
                            emo_text,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
                )

        with gr.Tab("Batch Generation"):
            gr.Markdown("Manage multiple prompt audios, give each its own text, generate in bulk, and retry specific entries as needed.")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        dataset_path_input = gr.Textbox(
                            label="Dataset train.txt path",
                            value="vivy_va_dataset/train.txt",
                            scale=3,
                            placeholder="Path to train.txt"
                        )
                        load_dataset_button = gr.Button("Load Dataset", scale=1)
                    batch_file_input = gr.Files(
                        label="Add prompt audio files",
                        file_types=["audio"],
                        file_count="multiple",
                        type="filepath"
                    )
                    batch_table = gr.Dataframe(
                        headers=["ID", "Prompt", "Text", "Output", "Status", "Last Generated"],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        row_count=(0, "dynamic"),
                        col_count=6,
                        interactive=False,
                        value=[]
                    )
                with gr.Column():
                    selected_entry = gr.Dropdown(label="Select entry", choices=[], value=None, interactive=True)
                    batch_prompt_player = gr.Audio(label="Prompt Audio", type="filepath", interactive=False)
                    batch_output_player = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
                    batch_text_input = gr.TextArea(label="Text", placeholder="Enter text for this entry", interactive=True)
                    batch_status = gr.Markdown(value="No entry selected.")
                    with gr.Row():
                        generate_all_button = gr.Button("Generate All")
                        regenerate_button = gr.Button("Regenerate Selected")
                    with gr.Row():
                        delete_entry_button = gr.Button("Delete Selected")
                        clear_entries_button = gr.Button("Clear All")

        with gr.Tab("Generation History"):
            gr.Markdown("View and manage your generated audio files from this session.")
            with gr.Row():
                history_gallery = gr.Gallery(
                    label="Generated Audio History",
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )
            with gr.Row():
                refresh_history_button = gr.Button("Refresh History", variant="secondary")
                clear_history_button = gr.Button("Clear All History", variant="stop")
            history_stats = gr.Markdown(value="No generations yet.")

        with gr.Tab("Compare Models"):
            gr.Markdown("Generate audio with two different models using the same prompt for side-by-side comparison.")

            # Model selection for comparison
            with gr.Row():
                with gr.Column(scale=1):
                    compare_model_a_dropdown = gr.Dropdown(
                        choices=[choice[0] for choice in model_choices_with_metadata],
                        value=model_choices_with_metadata[0][0] if len(model_choices_with_metadata) > 0 else None,
                        label="Model A",
                        interactive=True,
                        info="First model for comparison"
                    )
                    compare_model_a_info = gr.Markdown(value="Select Model A")

                with gr.Column(scale=1):
                    compare_model_b_dropdown = gr.Dropdown(
                        choices=[choice[0] for choice in model_choices_with_metadata],
                        value=model_choices_with_metadata[1][0] if len(model_choices_with_metadata) > 1 else None,
                        label="Model B",
                        interactive=True,
                        info="Second model for comparison"
                    )
                    compare_model_b_info = gr.Markdown(value="Select Model B")

            # Comparison inputs
            with gr.Row():
                with gr.Column(scale=1):
                    compare_prompt_audio = gr.Audio(
                        label="Voice Prompt",
                        sources=["upload", "microphone"],
                        type="filepath"
                    )
                with gr.Column(scale=2):
                    compare_text = gr.TextArea(
                        label="Text to Synthesize",
                        placeholder="Enter the same text for both models",
                        lines=3
                    )

            # Generate button and GPU selector
            with gr.Row():
                compare_gpu_dropdown = gr.Dropdown(
                    choices=gpu_choices,
                    value=gpu_choices[0][1] if gpu_choices else 0,
                    label="GPU for Comparison",
                    interactive=True,
                    scale=2
                )
                compare_generate_button = gr.Button(
                    "Generate with Both Models",
                    variant="primary",
                    scale=1
                )

            # Status and progress
            compare_status = gr.Markdown(value="Ready to compare models")

            # Results: Side-by-side audio players
            gr.Markdown("### Results")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Model A Output**")
                    compare_audio_a = gr.Audio(label="Generated Audio A", visible=False)
                with gr.Column(scale=1):
                    gr.Markdown("**Model B Output**")
                    compare_audio_b = gr.Audio(label="Generated Audio B", visible=False)

            # Waveform comparison visualization
            compare_waveform_image = gr.Image(
                label="Waveform Comparison",
                visible=False,
                type="filepath"
            )

            # Metrics comparison table
            compare_metrics_display = gr.Markdown(
                value="",
                visible=False
            )

        with gr.Tab("Training Setup"):
            gr.Markdown("""
            ### 🎓 Train New Language Model

            Configure and train a new TTS model for a specific language or use case.
            All files will be automatically named and organized.
            """)

            # Project configuration
            with gr.Accordion("Project Configuration", open=True):
                with gr.Row():
                    with gr.Column(scale=2):
                        train_project_name = gr.Textbox(
                            label="Project/Language Name",
                            placeholder="e.g., catalan, french, multilingual_romance",
                            info="Used for naming all output files (gpt_{name}.pth, {name}_bpe.model)",
                            value=""
                        )
                    with gr.Column(scale=1):
                        train_vocab_size = gr.Number(
                            label="Vocabulary Size",
                            value=12000,
                            precision=0,
                            info="Tokenizer vocabulary size"
                        )

                with gr.Row():
                    train_description = gr.Textbox(
                        label="Description (optional)",
                        placeholder="Brief description of this model (e.g., 'Catalan TTS with Barcelona accent')",
                        lines=2
                    )

                # Display computed paths
                train_paths_display = gr.Markdown(
                    value="**Enter a project name to see computed paths**",
                    visible=True
                )

            # Dataset configuration
            with gr.Accordion("Dataset Configuration", open=True):
                with gr.Row():
                    train_manifest = gr.File(
                        label="Training Manifest (JSONL)",
                        file_types=[".jsonl"],
                        type="filepath"
                    )
                    train_val_manifest = gr.File(
                        label="Validation Manifest (JSONL)",
                        file_types=[".jsonl"],
                        type="filepath"
                    )

                train_dataset_info = gr.Markdown(
                    value="Upload paired manifests (use tools/build_gpt_prompt_pairs.py to create them)"
                )

            # Tokenizer training
            with gr.Accordion("Step 1: Train Tokenizer", open=False):
                gr.Markdown("Train a BPE tokenizer on your text data")

                with gr.Row():
                    train_tokenizer_manifest = gr.File(
                        label="Raw Text Manifest (JSONL)",
                        file_types=[".jsonl"],
                        type="filepath",
                        info="Manifest with 'text' field (before pairing)"
                    )

                with gr.Row():
                    train_char_coverage = gr.Slider(
                        label="Character Coverage",
                        minimum=0.90,
                        maximum=1.0,
                        value=0.9995,
                        step=0.0001,
                        info="Keep near 1.0 for languages with many characters (e.g., Chinese, Japanese)"
                    )

                with gr.Row():
                    train_tokenizer_button = gr.Button(
                        "🔤 Train Tokenizer",
                        variant="primary",
                        size="lg"
                    )

                train_tokenizer_status = gr.Markdown(value="Ready to train tokenizer")
                train_tokenizer_output = gr.Textbox(
                    label="Tokenizer Training Log",
                    lines=10,
                    max_lines=20,
                    visible=False
                )

            # Model training
            with gr.Accordion("Step 2: Fine-tune Model", open=False):
                gr.Markdown("Fine-tune the GPT model on your language")

                with gr.Row():
                    with gr.Column():
                        train_base_checkpoint = gr.Dropdown(
                            choices=[Path(p).name for p in gpt_checkpoints],
                            value=Path(gpt_checkpoints[0]).name if gpt_checkpoints else None,
                            label="Base Checkpoint",
                            info="Starting point for fine-tuning"
                        )

                    with gr.Column():
                        train_gpu_id = gr.Dropdown(
                            choices=[(f"GPU {i}", i) for i in range(torch.cuda.device_count())],
                            value=0,
                            label="Training GPU",
                            info="GPU to use for training"
                        )

                with gr.Row():
                    with gr.Column():
                        train_batch_size = gr.Number(
                            label="Batch Size",
                            value=4,
                            precision=0,
                            info="Reduce if you get OOM errors"
                        )
                        train_epochs = gr.Number(
                            label="Epochs",
                            value=10,
                            precision=0,
                            info="Number of training epochs"
                        )

                    with gr.Column():
                        train_learning_rate = gr.Number(
                            label="Learning Rate",
                            value=2e-5,
                            info="Recommended: 1e-5 to 5e-5"
                        )
                        train_warmup_steps = gr.Number(
                            label="Warmup Steps",
                            value=500,
                            precision=0,
                            info="LR warmup steps"
                        )

                with gr.Row():
                    train_use_amp = gr.Checkbox(
                        label="Use Mixed Precision (AMP)",
                        value=True,
                        info="Faster training with FP16"
                    )

                with gr.Row():
                    train_model_button = gr.Button(
                        "🚀 Start Training",
                        variant="primary",
                        size="lg"
                    )
                    train_stop_button = gr.Button(
                        "⏹️ Stop Training",
                        variant="stop",
                        size="lg"
                    )

                train_model_status = gr.Markdown(value="Ready to start training")
                train_model_output = gr.Textbox(
                    label="Training Log",
                    lines=15,
                    max_lines=30,
                    visible=False
                )

            # Model installation
            with gr.Accordion("Step 3: Install Trained Model", open=False):
                gr.Markdown("Install the best checkpoint for use in WebUI")

                with gr.Row():
                    train_checkpoint_selector = gr.Dropdown(
                        choices=[],
                        label="Select Checkpoint",
                        info="Choose the best checkpoint to install",
                        interactive=True
                    )
                    train_refresh_checkpoints_button = gr.Button(
                        "🔄 Refresh",
                        size="sm"
                    )

                train_checkpoint_info = gr.Markdown(value="No checkpoints found yet")

                with gr.Row():
                    train_install_button = gr.Button(
                        "📦 Install Model to WebUI",
                        variant="primary",
                        size="lg"
                    )

                train_install_status = gr.Markdown(value="Ready to install")

        # Training Monitor Tab
        with gr.Tab("Training Monitor"):
            gr.Markdown("""
            ### 📊 Training Visualization & Monitoring

            Monitor your training progress in real-time with TensorBoard and live metrics.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    # TensorBoard Controls
                    with gr.Group():
                        gr.Markdown("#### TensorBoard")

                        with gr.Row():
                            monitor_project_selector = gr.Dropdown(
                                choices=[],
                                label="Select Project",
                                info="Choose a training project to monitor",
                                scale=3
                            )
                            monitor_refresh_projects = gr.Button("🔄 Refresh", scale=1)

                        with gr.Row():
                            monitor_tb_port = gr.Number(
                                label="TensorBoard Port",
                                value=6006,
                                precision=0,
                                scale=2
                            )
                            monitor_tb_start = gr.Button("▶️ Start TensorBoard", variant="primary", scale=2)
                            monitor_tb_stop = gr.Button("⏹️ Stop TensorBoard", scale=2)

                        monitor_tb_status = gr.Markdown(value="⚪ TensorBoard: Not running")

                        # TensorBoard iframe (hidden by default)
                        monitor_tb_frame = gr.HTML(
                            value="<p style='text-align: center; color: gray;'>Start TensorBoard to view</p>",
                            label="TensorBoard"
                        )

                with gr.Column(scale=1):
                    # Training Status
                    with gr.Group():
                        gr.Markdown("#### Training Status")
                        monitor_status_display = gr.Markdown(value="No training active")

                        monitor_refresh_status = gr.Button("🔄 Refresh Status")

            # Live Metrics Plots
            with gr.Accordion("📈 Live Metrics", open=True):
                with gr.Row():
                    with gr.Column():
                        monitor_loss_plot = gr.Plot(label="Training Loss")
                    with gr.Column():
                        monitor_lr_plot = gr.Plot(label="Learning Rate")

                with gr.Row():
                    monitor_refresh_plots = gr.Button("🔄 Refresh Plots")
                    monitor_auto_refresh = gr.Checkbox(
                        label="Auto-refresh (every 10s)",
                        value=False
                    )

                with gr.Row():
                    monitor_export_loss = gr.Button("💾 Export Loss Plot", size="sm")
                    monitor_export_lr = gr.Button("💾 Export LR Plot", size="sm")

                monitor_export_status = gr.Markdown(value="")

            # Alerts & Analysis
            with gr.Accordion("🚨 Alerts & Analysis", open=False):
                with gr.Row():
                    monitor_target_step = gr.Number(
                        label="Target Steps",
                        value=10000,
                        precision=0,
                        info="For time estimation"
                    )
                    monitor_analyze_button = gr.Button("🔍 Analyze Training", variant="primary")

                monitor_alerts_display = gr.Markdown(value="Click 'Analyze Training' to check for issues")

            # Run Comparison
            with gr.Accordion("📊 Compare Runs", open=False):
                gr.Markdown("Compare multiple training runs to find the best configuration")

                with gr.Row():
                    monitor_compare_run1 = gr.Dropdown(
                        choices=[],
                        label="Run 1",
                        scale=2
                    )
                    monitor_compare_run2 = gr.Dropdown(
                        choices=[],
                        label="Run 2",
                        scale=2
                    )
                    monitor_compare_runs_button = gr.Button("⚖️ Compare", scale=1)

                monitor_comparison_plot = gr.Plot(label="Loss Comparison")
                monitor_comparison_stats = gr.Markdown(value="")

            # Experiment Tracking Integration
            with gr.Accordion("🔗 Experiment Tracking", open=False):
                gr.Markdown("""
                **Optional:** Sync metrics to external tracking platforms

                - **W&B (Weights & Biases):** Cloud-based experiment tracking
                - **MLflow:** Self-hosted experiment tracking
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Weights & Biases")
                        monitor_wandb_project = gr.Textbox(
                            label="W&B Project Name",
                            placeholder="my-tts-project"
                        )
                        monitor_wandb_enabled = gr.Checkbox(
                            label="Enable W&B logging",
                            value=False
                        )
                        monitor_wandb_status = gr.Markdown(value="")

                    with gr.Column():
                        gr.Markdown("#### MLflow")
                        monitor_mlflow_run = gr.Textbox(
                            label="MLflow Run Name",
                            placeholder="catalan_experiment_1"
                        )
                        monitor_mlflow_enabled = gr.Checkbox(
                            label="Enable MLflow logging",
                            value=False
                        )
                        monitor_mlflow_status = gr.Markdown(value="")

    # Handler functions
    def handle_model_selection_change(model_label, gpt_paths_mapping):
        """Update metadata displays when model selection changes."""
        if not model_label or not gpt_paths_mapping:
            return "**Tokenizer:** No model selected", "**VRAM:** No model selected"

        gpt_path = gpt_paths_mapping.get(model_label)
        if not gpt_path:
            return "**Tokenizer:** Invalid selection", "**VRAM:** Invalid selection"

        tokenizer_info, vram_info = get_model_info_display(gpt_path)
        return tokenizer_info, vram_info

    def handle_model_load(model_label, gpu_id, gpt_paths_mapping):
        """Load selected model with specified GPU."""
        global _current_gpu_id

        if not model_label or not gpt_paths_mapping:
            gr.Warning("Select a model first.")
            return "⚠️ No model selected"

        gpt_path = gpt_paths_mapping.get(model_label)
        if not gpt_path:
            gr.Warning("Invalid model selection.")
            return "⚠️ Invalid selection"

        try:
            # Update GPU selection
            _current_gpu_id = gpu_id

            # Apply architecture-specific optimizations for the selected GPU
            # This ensures TF32, cuDNN, and other settings are optimal for the target GPU
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                opt_result = GPUConfig.apply_optimal_settings(gpu_id)
                if opt_result.get("status") == "applied":
                    logger.info(f"Applied {opt_result['architecture']} optimizations for GPU {gpu_id}")
                    # Set PyTorch default device
                    torch.cuda.set_device(gpu_id)

            # ModelManager will auto-detect tokenizer
            # We pass None for bpe_path since load_primary_tts uses ModelManager
            # which extracts and uses the tokenizer from metadata
            load_primary_tts(gpt_path, None)

            metadata = _model_manager.get_current_metadata()
            tokenizer_name = Path(metadata.tokenizer_path).name if metadata.tokenizer_path else "None"

            gr.Info(f"Model loaded successfully on GPU {gpu_id}!")
            return f"✅ Loaded: **{metadata.filename}** on GPU {gpu_id} | Tokenizer: **{tokenizer_name}**"
        except Exception as e:
            logger.exception("Failed to load model")
            gr.Warning(f"Failed to load model: {e}")
            return f"❌ Load failed: {e}"

    def refresh_monitor():
        """Refresh GPU monitor."""
        return get_gpu_monitor_text()

    # Emotion preset handlers
    def apply_emotion_preset(preset_id):
        """Apply emotion preset to sliders."""
        if preset_id == "custom":
            return [gr.update()] * 8  # No change for custom

        vector = get_preset_vector(preset_id)
        if vector is None:
            return [gr.update()] * 8

        # Return updates for all 8 sliders
        return [gr.update(value=v) for v in vector]

    def apply_emotion_mix(preset_a, preset_b, ratio):
        """Apply mixed emotion to sliders."""
        # Convert ratio from 0-100 to 0.0-1.0
        ratio_normalized = ratio / 100.0
        mixed_vector = mix_emotions(preset_a, preset_b, ratio_normalized)

        # Return updates for all 8 sliders
        return [gr.update(value=v) for v in mixed_vector]

    def update_preset_description(preset_id):
        """Update preset description display."""
        return get_preset_description(preset_id)

    # Duration estimation handler
    def update_duration_estimate(text, speech_rate="normal"):
        """Update duration estimation display."""
        if not text or len(text.strip()) == 0:
            return "📊 Enter text to see duration estimate"

        return get_duration_display(text, speech_rate)

    def handle_compare_model_info_update(model_label, gpt_paths_mapping):
        """Update model info display for comparison."""
        if not model_label or not gpt_paths_mapping:
            return "Select a model"

        gpt_path = gpt_paths_mapping.get(model_label)
        if not gpt_path:
            return "Invalid model"

        try:
            metadata = _model_manager.extract_model_metadata(gpt_path)
            info = f"**{metadata.filename}**\n"
            info += f"Size: {metadata.size_mb/1024:.1f} GB | "
            info += f"Version: {metadata.version} | "
            info += f"VRAM: ~{metadata.recommended_vram_gb:.1f} GB"
            return info
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return "Error loading metadata"

    def handle_compare_generate(
        model_a_label, model_b_label, text, prompt_audio, gpu_id, gpt_paths_mapping
    ):
        """Generate audio with both models and return comparison results."""
        import os
        from datetime import datetime

        # Validate inputs
        if not model_a_label or not model_b_label:
            gr.Warning("Please select both Model A and Model B")
            return (
                "⚠️ Please select both models",
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False)
            )

        if not text or not prompt_audio:
            gr.Warning("Please provide both text and voice prompt")
            return (
                "⚠️ Please provide text and voice prompt",
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False)
            )

        if model_a_label == model_b_label:
            gr.Warning("Please select different models for comparison")
            return (
                "⚠️ Please select different models",
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False)
            )

        # Get model paths
        model_a_path = gpt_paths_mapping.get(model_a_label)
        model_b_path = gpt_paths_mapping.get(model_b_label)

        if not model_a_path or not model_b_path:
            gr.Warning("Invalid model selection")
            return (
                "⚠️ Invalid model selection",
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False)
            )

        try:
            # Apply architecture-specific optimizations for the selected GPU
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                opt_result = GPUConfig.apply_optimal_settings(gpu_id)
                if opt_result.get("status") == "applied":
                    logger.info(f"Applied {opt_result['architecture']} optimizations for comparison on GPU {gpu_id}")
                    torch.cuda.set_device(gpu_id)

            # Create output directory
            output_dir = os.path.join("outputs", "comparisons")
            os.makedirs(output_dir, exist_ok=True)

            # Run comparison
            gr.Info(f"Comparing {Path(model_a_path).name} vs {Path(model_b_path).name}...")

            result = _model_comparator.compare_models(
                model_a_path=model_a_path,
                model_b_path=model_b_path,
                text=text,
                prompt_audio=prompt_audio,
                output_dir=output_dir,
                gpu_id=gpu_id,
                use_fp16=cmd_args.is_fp16
            )

            # Generate waveform comparison
            waveform_path = os.path.join(output_dir, f"waveform_{result.timestamp}.png")
            _model_comparator.generate_waveform_comparison(
                result.audio_a_path,
                result.audio_b_path,
                output_path=waveform_path
            )

            # Format metrics table
            metrics_table = _model_comparator.format_metrics_table(result)

            # Prepare status message
            status = f"""✅ Comparison completed!

**Model A:** {result.model_a_metrics.model_name}
- RTF: {result.model_a_metrics.rtf:.3f}
- Total Time: {result.model_a_metrics.total_time:.2f}s
- VRAM: {result.model_a_metrics.vram_peak_gb:.2f} GB

**Model B:** {result.model_b_metrics.model_name}
- RTF: {result.model_b_metrics.rtf:.3f}
- Total Time: {result.model_b_metrics.total_time:.2f}s
- VRAM: {result.model_b_metrics.vram_peak_gb:.2f} GB
"""

            gr.Info("Comparison completed successfully!")

            return (
                status,
                gr.update(value=result.audio_a_path, visible=True),
                gr.update(value=result.audio_b_path, visible=True),
                gr.update(value=waveform_path, visible=True),
                gr.update(value=metrics_table, visible=True)
            )

        except Exception as e:
            logger.exception("Comparison failed")
            gr.Warning(f"Comparison failed: {e}")
            return (
                f"❌ Comparison failed: {e}",
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False)
            )

    def handle_train_project_name_change(project_name):
        """Update path displays when project name changes."""
        if not project_name or not project_name.strip():
            return "**Enter a project name to see computed paths**"

        name = project_name.strip().lower().replace(" ", "_")

        paths_info = f"""
**Computed Paths:**

📁 **Training Directory:** `training/{name}/`
- Tokenizer: `training/{name}/tokenizer/{name}_bpe.model`
- Checkpoints: `training/{name}/checkpoints/model_stepXXXX.pth`
- Logs: `training/{name}/checkpoints/runs/`

📦 **Final Model Files:**
- Model: `models/gpt_{name}.pth`
- Tokenizer: `models/{name}_bpe.model`
- Config: `models/{name}_config.yaml`

✅ Base models in `checkpoints/` will NOT be modified
"""
        return paths_info

    def handle_train_tokenizer(
        project_name, vocab_size, char_coverage, tokenizer_manifest
    ):
        """Train BPE tokenizer."""
        import subprocess
        from pathlib import Path

        if not project_name or not project_name.strip():
            gr.Warning("Please enter a project name")
            return "❌ No project name provided", gr.update(visible=False)

        if not tokenizer_manifest:
            gr.Warning("Please upload a raw text manifest")
            return "❌ No manifest provided", gr.update(visible=False)

        name = project_name.strip().lower().replace(" ", "_")
        tokenizer_dir = Path(f"training/{name}/tokenizer")
        tokenizer_prefix = tokenizer_dir / f"{name}_bpe"

        try:
            # Run tokenizer training script
            cmd = [
                "python", "tools/tokenizer/train_bpe.py",
                "--manifest", tokenizer_manifest,
                "--output-prefix", str(tokenizer_prefix),
                "--vocab-size", str(int(vocab_size)),
                "--character-coverage", str(char_coverage)
            ]

            status_msg = f"🔄 Training tokenizer for '{name}'...\n"
            status_msg += f"Vocab size: {int(vocab_size)}\n"
            status_msg += f"Character coverage: {char_coverage}\n"

            gr.Info(f"Starting tokenizer training for {name}...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                status_msg += f"\n✅ **Tokenizer trained successfully!**\n"
                status_msg += f"📍 Location: `{tokenizer_prefix}.model`\n"
                gr.Info(f"Tokenizer training completed for {name}!")
                return status_msg, gr.update(value=result.stdout, visible=True)
            else:
                status_msg += f"\n❌ **Training failed**\n"
                error_msg = result.stderr or result.stdout
                gr.Warning(f"Tokenizer training failed: {error_msg[:200]}")
                return status_msg, gr.update(value=error_msg, visible=True)

        except subprocess.TimeoutExpired:
            gr.Warning("Tokenizer training timed out (>10 minutes)")
            return "❌ Training timed out", gr.update(visible=False)
        except Exception as e:
            logger.exception("Tokenizer training error")
            gr.Warning(f"Training error: {str(e)}")
            return f"❌ Error: {str(e)}", gr.update(visible=False)

    def handle_train_model(
        project_name,
        train_manifest,
        val_manifest,
        base_checkpoint,
        gpu_id,
        batch_size,
        epochs,
        learning_rate,
        warmup_steps,
        use_amp,
        gpt_paths
    ):
        """Start model training."""
        import subprocess
        from pathlib import Path

        if not project_name or not project_name.strip():
            gr.Warning("Please enter a project name")
            return "❌ No project name provided", gr.update(visible=False)

        if not train_manifest or not val_manifest:
            gr.Warning("Please upload both training and validation manifests")
            return "❌ Manifests missing", gr.update(visible=False)

        name = project_name.strip().lower().replace(" ", "_")
        tokenizer_path = Path(f"training/{name}/tokenizer/{name}_bpe.model")

        if not tokenizer_path.exists():
            gr.Warning(f"Tokenizer not found at {tokenizer_path}. Train tokenizer first!")
            return f"❌ Tokenizer not found: {tokenizer_path}", gr.update(visible=False)

        base_ckpt_path = next((p for p in gpt_paths if Path(p).name == base_checkpoint), None)
        if not base_ckpt_path:
            gr.Warning(f"Base checkpoint not found: {base_checkpoint}")
            return "❌ Base checkpoint not found", gr.update(visible=False)

        output_dir = Path(f"training/{name}/checkpoints")

        try:
            # Build training command
            cmd = [
                "python", "trainers/train_gpt_v2.py",
                "--train-manifest", train_manifest,
                "--val-manifest", val_manifest,
                "--tokenizer", str(tokenizer_path),
                "--config", "checkpoints/config.yaml",
                "--base-checkpoint", base_ckpt_path,
                "--output-dir", str(output_dir),
                "--batch-size", str(int(batch_size)),
                "--epochs", str(int(epochs)),
                "--learning-rate", str(learning_rate),
                "--warmup-steps", str(int(warmup_steps)),
            ]

            if use_amp:
                cmd.append("--amp")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            status_msg = f"🚀 **Training started for '{name}'**\n\n"
            status_msg += f"📊 Configuration:\n"
            status_msg += f"- Base: {base_checkpoint}\n"
            status_msg += f"- Tokenizer: {tokenizer_path.name}\n"
            status_msg += f"- GPU: {gpu_id}\n"
            status_msg += f"- Batch size: {int(batch_size)}\n"
            status_msg += f"- Epochs: {int(epochs)}\n"
            status_msg += f"- Learning rate: {learning_rate}\n"
            status_msg += f"- Output: `{output_dir}/`\n\n"
            status_msg += "⏳ Training in progress... (this may take hours)\n"
            status_msg += "Check TensorBoard: `tensorboard --logdir " + str(output_dir / "runs") + "`\n"

            gr.Info(f"Training started for {name}! Check console for progress.")

            # Note: Running training in background would be better with a proper job queue
            # For now, we show the command and let user run it in terminal
            command_str = " ".join(cmd)
            log_msg = f"Run this command in a terminal:\n\nCUDA_VISIBLE_DEVICES={gpu_id} {command_str}\n"

            return status_msg, gr.update(value=log_msg, visible=True)

        except Exception as e:
            logger.exception("Training setup error")
            gr.Warning(f"Training setup error: {str(e)}")
            return f"❌ Error: {str(e)}", gr.update(visible=False)

    def handle_refresh_checkpoints(project_name):
        """Refresh checkpoint list."""
        from pathlib import Path

        if not project_name or not project_name.strip():
            return gr.update(choices=[]), "Enter a project name first"

        name = project_name.strip().lower().replace(" ", "_")
        checkpoint_dir = Path(f"training/{name}/checkpoints")

        if not checkpoint_dir.exists():
            return gr.update(choices=[]), f"No checkpoints found. Directory `{checkpoint_dir}` doesn't exist."

        # Find all checkpoint files
        checkpoints = list(checkpoint_dir.glob("model_step*.pth"))
        checkpoints.sort(key=lambda p: int(p.stem.replace("model_step", "")))

        if not checkpoints:
            return gr.update(choices=[]), "No checkpoints found yet. Start training first!"

        choices = [str(ckpt.relative_to(Path.cwd())) for ckpt in checkpoints]

        # Also add latest.pth if it exists
        latest_path = checkpoint_dir / "latest.pth"
        if latest_path.exists():
            choices.append(str(latest_path.relative_to(Path.cwd())))

        info_msg = f"**Found {len(checkpoints)} checkpoints:**\n"
        for ckpt in checkpoints[-5:]:  # Show last 5
            info_msg += f"- {ckpt.name}\n"

        return gr.update(choices=choices, value=choices[-1] if choices else None), info_msg

    def handle_install_model(project_name, checkpoint_path, description):
        """Install trained model to models/ directory."""
        import subprocess
        from pathlib import Path

        if not project_name or not project_name.strip():
            gr.Warning("Please enter a project name")
            return "❌ No project name provided"

        if not checkpoint_path:
            gr.Warning("Please select a checkpoint")
            return "❌ No checkpoint selected"

        name = project_name.strip().lower().replace(" ", "_")
        tokenizer_path = Path(f"training/{name}/tokenizer/{name}_bpe.model")

        if not tokenizer_path.exists():
            gr.Warning(f"Tokenizer not found at {tokenizer_path}")
            return f"❌ Tokenizer not found: {tokenizer_path}"

        try:
            # Run install script
            cmd = [
                "python", "tools/install_trained_model.py",
                "--checkpoint", checkpoint_path,
                "--tokenizer", str(tokenizer_path),
                "--output-name", name,
                "--output-dir", "models",
            ]

            if description:
                cmd.extend(["--description", description])

            cmd.append("--force")  # Auto-overwrite in UI mode

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                status_msg = f"✅ **Model installed successfully!**\n\n"
                status_msg += f"📦 **Installed as:**\n"
                status_msg += f"- `models/gpt_{name}.pth`\n"
                status_msg += f"- `models/{name}_bpe.model`\n\n"
                status_msg += "🎉 Your model is now available in the Model Checkpoint dropdown!\n"
                status_msg += "Reload the page to see it."

                gr.Info(f"Model {name} installed successfully!")
                return status_msg
            else:
                error_msg = result.stderr or result.stdout
                gr.Warning(f"Installation failed: {error_msg[:200]}")
                return f"❌ Installation failed:\n{error_msg}"

        except subprocess.TimeoutExpired:
            gr.Warning("Installation timed out")
            return "❌ Installation timed out"
        except Exception as e:
            logger.exception("Installation error")
            gr.Warning(f"Installation error: {str(e)}")
            return f"❌ Error: {str(e)}"

    def refresh_history_gallery():
        """Refresh history gallery."""
        history = get_history_manager()
        gallery_data = history.get_gallery_data(limit=50)
        stats = history.get_statistics()
        stats_text = f"""**Session Statistics:**
- Total Generations: {stats['total_generations']}
- Total Duration: {stats['total_duration']:.1f}s
- Average RTF: {stats['avg_rtf']:.3f}
"""
        return gallery_data, stats_text

    # Training Monitor Handlers
    def refresh_training_projects():
        """Refresh list of available training projects."""
        training_root = Path("training")
        if not training_root.exists():
            return gr.update(choices=[])

        # Find all project directories
        projects = []
        for project_dir in training_root.iterdir():
            if project_dir.is_dir():
                # Check if it has a checkpoints directory (indicates training)
                if (project_dir / "checkpoints").exists():
                    projects.append(project_dir.name)

        projects.sort()
        return gr.update(choices=projects, value=projects[0] if projects else None)

    def start_tensorboard(project_name, port):
        """Start TensorBoard for the selected project."""
        if not project_name:
            gr.Warning("Please select a project first")
            return (
                "⚠️ No project selected",
                "<p style='text-align: center; color: gray;'>Select a project and start TensorBoard</p>"
            )

        try:
            port = int(port)
        except (ValueError, TypeError):
            port = 6006

        logdir = get_tensorboard_logdir(project_name)

        success, message = _tensorboard_manager.start(logdir, port)

        if success:
            # Create iframe HTML
            iframe_html = f"""
            <iframe src="{message}"
                    width="100%"
                    height="800px"
                    frameborder="0">
            </iframe>
            """
            status = f"🟢 TensorBoard running at [{message}]({message})"
            gr.Info(f"TensorBoard started at {message}")
            return status, iframe_html
        else:
            gr.Warning(f"Failed to start TensorBoard: {message}")
            return (
                f"🔴 TensorBoard failed: {message}",
                f"<p style='text-align: center; color: red;'>Error: {message}</p>"
            )

    def stop_tensorboard():
        """Stop TensorBoard."""
        success, message = _tensorboard_manager.stop()

        if success:
            gr.Info("TensorBoard stopped")
            return (
                "⚪ TensorBoard: Not running",
                "<p style='text-align: center; color: gray;'>TensorBoard stopped</p>"
            )
        else:
            return (
                f"⚠️ {message}",
                "<p style='text-align: center; color: gray;'>TensorBoard not running</p>"
            )

    def refresh_training_status(project_name):
        """Refresh training status display."""
        if not project_name:
            return "⚠️ No project selected"

        training_root = Path("training")
        project_dir = training_root / project_name / "checkpoints"

        if not project_dir.exists():
            return f"⚠️ Project directory not found: {project_dir}"

        # Find log files
        log_files = find_training_logs(project_name)

        if not log_files:
            return f"📂 **Project:** {project_name}\n\n⚠️ No training logs found yet.\n\nStart training to see metrics."

        # Parse most recent log
        parser = TrainingLogParser(log_files[0])
        latest_metrics = parser.get_latest_metrics()

        if not latest_metrics:
            return f"📂 **Project:** {project_name}\n\n⚠️ No metrics parsed yet from logs."

        # Format status
        status = f"""📂 **Project:** `{project_name}`

📊 **Latest Metrics:**
- Step: {latest_metrics.step}
- Epoch: {latest_metrics.epoch}
- Loss: {latest_metrics.loss:.4f}
- Learning Rate: {latest_metrics.learning_rate:.2e}
"""

        if latest_metrics.grad_norm:
            status += f"- Grad Norm: {latest_metrics.grad_norm:.4f}\n"
        if latest_metrics.vram_gb:
            status += f"- VRAM: {latest_metrics.vram_gb:.1f} GB\n"

        # Find latest checkpoint
        checkpoints = list(project_dir.glob("model_step*.pth"))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            status += f"\n💾 **Latest Checkpoint:** `{latest_ckpt.name}`"

        return status

    def refresh_training_plots(project_name):
        """Refresh training metric plots."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not project_name:
            # Return empty plots
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No project selected",
                xaxis_title="Step",
                yaxis_title="Value"
            )
            return empty_fig, empty_fig

        # Find log files
        log_files = find_training_logs(project_name)

        if not log_files:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No training logs found",
                xaxis_title="Step",
                yaxis_title="Value"
            )
            return empty_fig, empty_fig

        # Parse log file
        parser = TrainingLogParser(log_files[0])
        metrics = parser.get_metrics_for_plotting()

        if not metrics['steps']:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No metrics found in logs",
                xaxis_title="Step",
                yaxis_title="Value"
            )
            return empty_fig, empty_fig

        # Loss plot
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=metrics['steps'],
            y=metrics['losses'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#2563eb', width=2),
            marker=dict(size=4)
        ))
        loss_fig.update_layout(
            title=f"Training Loss - {project_name}",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            template="plotly_white",
            hovermode='x unified'
        )

        # Learning rate plot
        lr_fig = go.Figure()
        lr_fig.add_trace(go.Scatter(
            x=metrics['steps'],
            y=metrics['learning_rates'],
            mode='lines+markers',
            name='Learning Rate',
            line=dict(color='#16a34a', width=2),
            marker=dict(size=4)
        ))
        lr_fig.update_layout(
            title=f"Learning Rate - {project_name}",
            xaxis_title="Training Step",
            yaxis_title="Learning Rate",
            template="plotly_white",
            hovermode='x unified'
        )

        return loss_fig, lr_fig

    def export_plot(project_name, plot_type):
        """Export plot to PNG file."""
        if not project_name:
            return "⚠️ No project selected"

        # Generate plots
        loss_fig, lr_fig = refresh_training_plots(project_name)

        # Choose which plot to export
        fig = loss_fig if plot_type == "loss" else lr_fig

        # Create export directory
        export_dir = Path("outputs") / "training_plots"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = export_dir / f"{project_name}_{plot_type}_{timestamp}.png"

        success = export_plot_to_png(fig, output_path)

        if success:
            gr.Info(f"Plot exported to {output_path}")
            return f"✅ Exported to: `{output_path}`"
        else:
            return "❌ Export failed (install kaleido: pip install kaleido)"

    def analyze_training(project_name, target_step):
        """Analyze training for issues and estimate time."""
        if not project_name:
            return "⚠️ No project selected"

        # Find and parse logs
        log_files = find_training_logs(project_name)
        if not log_files:
            return "⚠️ No training logs found"

        parser = TrainingLogParser(log_files[0])
        metrics = parser.parse_log_file()

        if not metrics:
            return "⚠️ No metrics found in logs"

        # Run analyses
        analysis = "## 🔍 Training Analysis\n\n"

        # 1. Plateau detection
        is_plateau, plateau_msg = _training_analyzer.detect_plateau(metrics, patience=50)
        if is_plateau:
            analysis += f"### ⚠️ Plateau Alert\n{plateau_msg}\n\n"
        else:
            analysis += f"### ✅ Training Progress\n{plateau_msg}\n\n"

        # 2. Divergence detection
        is_diverging, div_msg = _training_analyzer.detect_divergence(metrics, threshold=10.0)
        if is_diverging:
            analysis += f"### 🔴 Divergence Alert\n{div_msg}\n\n"
        else:
            analysis += f"### ✅ Stability Check\n{div_msg}\n\n"

        # 3. Time estimation
        seconds, time_msg = _training_analyzer.estimate_time_remaining(metrics, int(target_step))
        analysis += f"### {time_msg}\n\n"

        # 4. Current stats
        latest = metrics[-1]
        analysis += f"### 📊 Current Stats\n"
        analysis += f"- Current Step: {latest.step:,}\n"
        analysis += f"- Current Loss: {latest.loss:.4f}\n"
        analysis += f"- Learning Rate: {latest.learning_rate:.2e}\n"

        if latest.grad_norm:
            analysis += f"- Grad Norm: {latest.grad_norm:.4f}\n"

        # 5. Recommendations
        analysis += f"\n### 💡 Recommendations\n"
        if is_plateau:
            analysis += "- Consider reducing learning rate\n"
            analysis += "- Check if you've reached model capacity\n"
            analysis += "- Try adjusting batch size\n"
        if is_diverging:
            analysis += "- **URGENT:** Stop training immediately\n"
            analysis += "- Reduce learning rate significantly\n"
            analysis += "- Check for data quality issues\n"

        return analysis

    def compare_training_runs(run1_name, run2_name):
        """Compare two training runs."""
        import plotly.graph_objects as go

        if not run1_name or not run2_name:
            return go.Figure(), "⚠️ Select both runs to compare"

        if run1_name == run2_name:
            return go.Figure(), "⚠️ Please select different runs"

        # Parse both runs
        runs = {}
        for run_name in [run1_name, run2_name]:
            log_files = find_training_logs(run_name)
            if log_files:
                parser = TrainingLogParser(log_files[0])
                runs[run_name] = parser.parse_log_file()

        if len(runs) != 2:
            return go.Figure(), "⚠️ Could not load logs for both runs"

        # Create comparison plot
        fig = go.Figure()

        colors = ['#2563eb', '#dc2626']
        for i, (run_name, metrics) in enumerate(runs.items()):
            steps = [m.step for m in metrics]
            losses = [m.loss for m in metrics]

            fig.add_trace(go.Scatter(
                x=steps,
                y=losses,
                mode='lines+markers',
                name=run_name,
                line=dict(color=colors[i], width=2),
                marker=dict(size=4)
            ))

        fig.update_layout(
            title="Training Loss Comparison",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            template="plotly_white",
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )

        # Get comparison stats
        comparison = _training_analyzer.compare_runs(runs)

        stats = f"## 📊 Run Comparison\n\n"

        for run_name, run_stats in comparison.items():
            if run_name.startswith('_'):
                continue

            stats += f"### {run_name}\n"
            stats += f"- Best Loss: {run_stats['best_loss']:.4f}\n"
            stats += f"- Final Loss: {run_stats['final_loss']:.4f}\n"
            stats += f"- Average Loss: {run_stats['avg_loss']:.4f}\n"
            stats += f"- Total Steps: {run_stats['total_steps']:,}\n"
            stats += f"- Final LR: {run_stats['final_lr']:.2e}\n\n"

        stats += f"### 🏆 Winner\n"
        stats += f"**{comparison['_best_run']}** with best loss of **{comparison['_best_loss']:.4f}**"

        return fig, stats

    def clear_history_all():
        """Clear all history."""
        history = get_history_manager()
        history.clear_history()
        gr.Info("History cleared successfully.")
        return [], "No generations yet."

    def on_input_text_change(text, max_tokens_per_sentence):
        if text and len(text) > 0:
            try:
                current_tts = ensure_primary_tts()
                text_tokens_list = current_tts.tokenizer.tokenize(text)

                sentences = current_tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_tokens_per_sentence))
                data = []
                for i, s in enumerate(sentences):
                    sentence_str = ''.join(s)
                    tokens_count = len(s)
                    data.append([i, sentence_str, tokens_count])
                return {
                    sentences_preview: gr.update(value=data, visible=True, type="array"),
                }
            except RuntimeError:
                return {
                    sentences_preview: gr.update(value=[], visible=True, type="array"),
                }
        else:
            return {
                sentences_preview: gr.update(value=[], visible=True, type="array"),
            }
    def on_method_select(emo_control_method):
        if emo_control_method == 1:
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 2:
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 3:
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        else:
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    def build_batch_table_data(rows):
        table_data = []
        for row in rows or []:
            text_preview = row.get("text", "") or ""
            if text_preview and len(text_preview) > 60:
                text_preview = text_preview[:57] + "..."
            table_data.append([
                row.get("id"),
                os.path.basename(row.get("prompt_path", "")) if row.get("prompt_path") else "",
                text_preview,
                os.path.basename(row.get("output_path", "")) if row.get("output_path") else "",
                row.get("status", "Pending"),
                row.get("last_generated", "")
            ])
        return table_data

    def find_batch_row(rows, row_id):
        if row_id is None:
            return None
        for row in rows or []:
            if row.get("id") == row_id:
                return row
        return None

    def resolve_batch_selection(rows, selected_value):
        choices = [str(row.get("id")) for row in rows or []]
        if not choices:
            return gr.update(choices=[], value=None), None
        if selected_value is not None:
            selected_str = str(selected_value)
            if selected_str in choices:
                return gr.update(choices=choices, value=selected_str), int(selected_str)
        selected_str = choices[-1]
        return gr.update(choices=choices, value=selected_str), int(selected_str)

    def prepare_batch_selection(rows, selected_value):
        dropdown_update, resolved_id = resolve_batch_selection(rows, selected_value)
        row = find_batch_row(rows, resolved_id)
        prompt_update = gr.update(value=row.get("prompt_path") if row else None)
        output_update = gr.update(value=row.get("output_path") if row and row.get("output_path") else None)
        text_update = gr.update(value=row.get("text", "") if row else "")
        return dropdown_update, resolved_id, prompt_update, output_update, text_update, row

    def format_batch_status(row, message=None):
        if not row:
            base = "No entry selected."
        else:
            lines = [f"Row {row.get('id')}: {row.get('status', 'Pending')}"]
            if row.get("text"):
                preview = row.get("text")
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                lines.append(f"Text: {preview}")
            if row.get("output_path"):
                lines.append(f"Output: {row.get('output_path')}")
            if row.get("last_generated"):
                lines.append(f"Last generated: {row.get('last_generated')}")
            base = "\n".join(lines)
        if message:
            if base:
                base = f"{base}\n{message}"
            else:
                base = message
        return gr.update(value=base)

    def add_batch_prompts(files, rows, next_id, selected_value):
        rows = rows or []
        next_id = next_id or 1
        files = files or []
        updated_rows = [dict(row) for row in rows]
        prompts_dir = os.path.abspath(os.path.join(current_dir, "prompts"))
        os.makedirs(prompts_dir, exist_ok=True)
        added = 0
        last_added_id = None
        for file_path in files:
            if not file_path:
                continue
            safe_name = os.path.basename(file_path)
            timestamp = int(time.time() * 1000)
            target_name = f"batch_prompt_{next_id}_{timestamp}_{safe_name}"
            target_path = os.path.join(prompts_dir, target_name)
            try:
                shutil.copy(file_path, target_path)
            except Exception as exc:
                logger.exception("Failed to store prompt %s", file_path)
                gr.Warning(f"Failed to add {safe_name}: {exc}")
                continue
            entry = {
                "id": next_id,
                "prompt_path": target_path,
                "output_path": None,
                "status": "Pending",
                "last_generated": "",
                "text": "",
            }
            updated_rows.append(entry)
            added += 1
            last_added_id = entry["id"]
            next_id += 1
        selected_seed = last_added_id if added else selected_value
        dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(updated_rows, selected_seed)
        table_update = gr.update(value=build_batch_table_data(updated_rows))
        status_message = None
        if added:
            status_message = f"Added {added} prompt{'s' if added != 1 else ''}."
        elif files:
            status_message = "No new prompts were added."
        status_update = format_batch_status(selected_row, status_message)
        return updated_rows, next_id, gr.update(value=None), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def load_dataset_entries(dataset_path, rows, next_id, selected_value, progress=gr.Progress()):
        rows = rows or []
        next_id = next_id or 1
        dataset_path = (dataset_path or "").strip()
        if not dataset_path:
            gr.Warning("Provide a dataset train.txt path before loading.")
            dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row)
            return rows, next_id, gr.update(value=""), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        dataset_path_abs = dataset_path if os.path.isabs(dataset_path) else os.path.abspath(os.path.join(current_dir, dataset_path))
        if not os.path.exists(dataset_path_abs):
            gr.Warning(f"Dataset file not found: {dataset_path_abs}")
            dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row)
            return rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        dataset_dir = os.path.dirname(dataset_path_abs)
        audio_dirs = [dataset_dir, os.path.join(dataset_dir, "wavs"), os.path.join(dataset_dir, "audio")]

        try:
            lines = Path(dataset_path_abs).read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            gr.Warning(f"Failed to read dataset file: {exc}")
            dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row)
            return rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        updated_rows = [dict(row) for row in rows]
        prompts_dir = os.path.abspath(os.path.join(current_dir, "prompts"))
        os.makedirs(prompts_dir, exist_ok=True)

        existing_prompts = {os.path.basename(r.get("prompt_path", "")) for r in updated_rows if r.get("prompt_path")}

        total_lines = len(lines)
        added = 0
        skipped_missing_audio = 0
        skipped_invalid = 0

        progress(0.0, desc="Parsing dataset")
        for idx, line in enumerate(lines):
            progress(min((idx + 1) / max(total_lines, 1), 0.95), desc=f"Processing line {idx + 1}/{total_lines}")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("|", 1)
            if len(parts) != 2:
                skipped_invalid += 1
                continue
            audio_name = parts[0].strip()
            text_value = parts[1].strip()
            if not audio_name or not text_value:
                skipped_invalid += 1
                continue

            audio_source = None
            for base_dir in audio_dirs:
                candidate = os.path.join(base_dir, audio_name)
                if os.path.exists(candidate):
                    audio_source = candidate
                    break
            if not audio_source:
                skipped_missing_audio += 1
                continue

            unique_suffix = f"dataset_{next_id}_{int(time.time() * 1000)}"
            target_name = f"{unique_suffix}_{os.path.basename(audio_name)}"
            if target_name in existing_prompts:
                target_name = f"{unique_suffix}_{next_id}_{os.path.basename(audio_name)}"
            target_path = os.path.join(prompts_dir, target_name)
            try:
                shutil.copy(audio_source, target_path)
            except Exception as exc:
                logger.exception("Failed to copy dataset prompt %s", audio_source)
                gr.Warning(f"Failed to copy {audio_name}: {exc}")
                skipped_missing_audio += 1
                continue

            entry = {
                "id": next_id,
                "prompt_path": target_path,
                "output_path": None,
                "status": "Pending",
                "last_generated": "",
                "text": text_value,
            }
            updated_rows.append(entry)
            existing_prompts.add(target_name)
            added += 1
            next_id += 1

        selected_seed = updated_rows[-1]["id"] if added else selected_value
        dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(updated_rows, selected_seed)
        table_update = gr.update(value=build_batch_table_data(updated_rows))

        status_parts = []
        if added:
            status_parts.append(f"Loaded {added} entries")
        if skipped_missing_audio:
            status_parts.append(f"{skipped_missing_audio} missing audio")
        if skipped_invalid:
            status_parts.append(f"{skipped_invalid} invalid lines")
        status_message = ", ".join(status_parts) if status_parts else "No new entries loaded."
        status_update = format_batch_status(selected_row, status_message)
        progress(1.0, desc="Dataset load complete")
        return updated_rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def validate_emotion_settings(emo_control_method, vec_values):
        mode = emo_control_method if isinstance(emo_control_method, int) else getattr(emo_control_method, "value", 0)
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            mode = 0
        vec = None
        if mode == 2:
            vec = vec_values
            if sum(vec_values) > 1.5:
                gr.Warning("Emotion vector sum cannot exceed 1.5. Adjust the sliders and retry.")
                return mode, None
        return mode, vec

    def build_generation_kwargs(do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens):
        try:
            top_k_value = int(top_k)
        except (TypeError, ValueError):
            top_k_value = 0
        try:
            num_beams_value = int(num_beams)
        except (TypeError, ValueError):
            num_beams_value = 1
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": top_k_value if top_k_value > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams_value,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        return kwargs

    def generate_all_batch(rows, selected_value, emo_control_method, emo_ref_path, emo_weight,
                           vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                           emo_text, emo_random, max_text_tokens_per_sentence,
                           do_sample, top_p, top_k, temperature,
                           length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                           progress=gr.Progress()):
        rows = rows or []
        if not rows:
            gr.Warning("Add prompt audio files before generating.")
            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row)
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        vec_values = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        mode, vec = validate_emotion_settings(emo_control_method, vec_values)
        if mode == 2 and vec is None:
            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row, "Emotion vector sum exceeded limit.")
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        try:
            max_tokens = int(max_text_tokens_per_sentence)
        except (TypeError, ValueError):
            max_tokens = 120

        generation_kwargs = build_generation_kwargs(do_sample, top_p, top_k, temperature,
                                                    length_penalty, num_beams, repetition_penalty, max_mel_tokens)

        outputs_dir = os.path.abspath(os.path.join(current_dir, "outputs", "tasks"))
        os.makedirs(outputs_dir, exist_ok=True)

        updated_rows = []
        total = len(rows)
        success_count = 0
        progress(0.0, desc="Starting batch generation")
        for idx, row in enumerate(rows):
            new_row = dict(row)
            prompt_path = new_row.get("prompt_path")
            if not prompt_path or not os.path.exists(prompt_path):
                new_row["status"] = "Error: Prompt missing"
                updated_rows.append(new_row)
                continue
            text_value = (new_row.get("text") or "").strip()
            if not text_value:
                new_row["status"] = "Error: Text missing"
                updated_rows.append(new_row)
                continue
            output_filename = f"batch_row_{new_row['id']}_{int(time.time() * 1000)}.wav"
            output_path = os.path.join(outputs_dir, output_filename)
            try:
                tts.gr_progress = progress
                tts.infer(
                    spk_audio_prompt=prompt_path,
                    text=text_value,
                    output_path=output_path,
                    emo_audio_prompt=emo_ref_path if mode == 1 else None,
                    emo_alpha=float(emo_weight) if mode == 1 else 1.0,
                    emo_vector=vec if mode == 2 else None,
                    use_emo_text=(mode == 3), emo_text=emo_text,
                    use_random=emo_random,
                    verbose=cmd_args.verbose,
                    max_text_tokens_per_sentence=max_tokens,
                    **generation_kwargs,
                )
                new_row["output_path"] = output_path
                new_row["status"] = "Completed"
                new_row["last_generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                success_count += 1
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    new_row["status"] = "Error: GPU OOM - Try reducing parameters"
                    logger.exception("GPU out of memory for row %s", new_row.get("id"))
                else:
                    logger.exception("Batch generation failed for row %s", new_row.get("id"))
                    new_row["status"] = f"Error: {exc}"
            except Exception as exc:
                logger.exception("Batch generation failed for row %s", new_row.get("id"))
                new_row["status"] = f"Error: {exc}"
            updated_rows.append(new_row)
            progress(min((idx + 1) / total, 1.0), desc=f"Generated {idx + 1}/{total}")

        # Clean up GPU memory after batch completion
        cleanup_gpu_memory()

        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(updated_rows, selected_value)
        table_update = gr.update(value=build_batch_table_data(updated_rows))
        status_message = f"Generated {success_count}/{total} entr{'ies' if total != 1 else 'y'}."
        status_update = format_batch_status(row, status_message)
        return updated_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def regenerate_batch_entry(rows, selected_value, emo_control_method, emo_ref_path, emo_weight,
                               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                               emo_text, emo_random, max_text_tokens_per_sentence,
                               do_sample, top_p, top_k, temperature,
                               length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                               progress=gr.Progress()):
        rows = rows or []
        dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(rows, selected_value)
        if not selected_row:
            gr.Warning("Select an entry to regenerate.")
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(None)
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        vec_values = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        mode, vec = validate_emotion_settings(emo_control_method, vec_values)
        if mode == 2 and vec is None:
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(selected_row, "Emotion vector sum exceeded limit.")
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        try:
            max_tokens = int(max_text_tokens_per_sentence)
        except (TypeError, ValueError):
            max_tokens = 120

        generation_kwargs = build_generation_kwargs(do_sample, top_p, top_k, temperature,
                                                    length_penalty, num_beams, repetition_penalty, max_mel_tokens)

        prompt_path = selected_row.get("prompt_path")
        if not prompt_path or not os.path.exists(prompt_path):
            gr.Warning("Prompt audio file is missing.")
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(selected_row, "Prompt audio file missing.")
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        text_value = (selected_row.get("text") or "").strip()
        if not text_value:
            gr.Warning("Enter text for this entry before regenerating.")
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(selected_row, "Text is missing.")
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        outputs_dir = os.path.abspath(os.path.join(current_dir, "outputs", "tasks"))
        os.makedirs(outputs_dir, exist_ok=True)
        output_filename = f"batch_row_{selected_row['id']}_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(outputs_dir, output_filename)
        result_rows = []
        for row in rows:
            if row.get("id") != selected_row.get("id"):
                result_rows.append(dict(row))
                continue
            updated_row = dict(row)
            try:
                tts.gr_progress = progress
                tts.infer(
                    spk_audio_prompt=prompt_path,
                    text=text_value,
                    output_path=output_path,
                    emo_audio_prompt=emo_ref_path if mode == 1 else None,
                    emo_alpha=float(emo_weight) if mode == 1 else 1.0,
                    emo_vector=vec if mode == 2 else None,
                    use_emo_text=(mode == 3), emo_text=emo_text,
                    use_random=emo_random,
                    verbose=cmd_args.verbose,
                    max_text_tokens_per_sentence=max_tokens,
                    **generation_kwargs,
                )
                updated_row["output_path"] = output_path
                updated_row["status"] = "Completed"
                updated_row["last_generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    updated_row["status"] = "Error: GPU OOM - Try reducing parameters"
                    logger.exception("GPU out of memory for row %s", updated_row.get("id"))
                else:
                    logger.exception("Regeneration failed for row %s", updated_row.get("id"))
                    updated_row["status"] = f"Error: {exc}"
            except Exception as exc:
                logger.exception("Regeneration failed for row %s", updated_row.get("id"))
                updated_row["status"] = f"Error: {exc}"
            result_rows.append(updated_row)

        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(result_rows, selected_value)
        table_update = gr.update(value=build_batch_table_data(result_rows))
        status_update = format_batch_status(row, "Regeneration finished.")
        return result_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def delete_batch_entry(rows, selected_value):
        rows = rows or []
        dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(rows, selected_value)
        if not selected_row:
            gr.Warning("Select an entry to delete.")
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(None)
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update
        remaining_rows = [dict(row) for row in rows if row.get("id") != selected_row.get("id")]
        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(remaining_rows, None)
        table_update = gr.update(value=build_batch_table_data(remaining_rows))
        status_update = format_batch_status(row, "Entry deleted.")
        return remaining_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def clear_batch_rows(rows, next_id):
        dropdown_update = gr.update(choices=[], value=None)
        prompt_update = gr.update(value=None)
        output_update = gr.update(value=None)
        text_update = gr.update(value="")
        status_update = format_batch_status(None, "Batch list cleared.")
        return [], 1, gr.update(value=[]), dropdown_update, prompt_update, output_update, text_update, status_update

    def on_select_batch_entry(selected_value, rows):
        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
        status_update = format_batch_status(row)
        return dropdown_update, prompt_update, output_update, text_update, status_update

    def update_batch_text(new_text, rows, selected_value):
        rows = rows or []
        try:
            selected_id = int(selected_value) if selected_value is not None else None
        except (TypeError, ValueError):
            selected_id = None

        if selected_id is None:
            gr.Warning("Select an entry before editing text.")
            table_update = gr.update(value=build_batch_table_data(rows))
            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            status_update = format_batch_status(row)
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        updated_rows = []
        target_row = None
        for row in rows:
            row_id = row.get("id")
            new_row = dict(row)
            if row_id == selected_id:
                new_row["text"] = new_text
                if new_row.get("output_path"):
                    new_row["status"] = "Pending"
                target_row = new_row
            updated_rows.append(new_row)

        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(updated_rows, selected_id)
        table_update = gr.update(value=build_batch_table_data(updated_rows))
        status_message = "Text updated. Regenerate to apply." if target_row else ""
        status_update = format_batch_status(row, status_message if status_message else None)
        return updated_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    emo_control_method.select(on_method_select,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emo_random,
                 emotion_vector_group,
                 emo_text_group]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    max_text_tokens_per_sentence.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_sentence,
                             target_duration,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

    # Emotion preset connections
    apply_preset_btn.click(
        apply_emotion_preset,
        inputs=[emotion_preset],
        outputs=[vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    apply_mix_btn.click(
        apply_emotion_mix,
        inputs=[mix_preset_a, mix_preset_b, mix_ratio],
        outputs=[vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    emotion_preset.change(
        update_preset_description,
        inputs=[emotion_preset],
        outputs=[preset_description]
    )

    # Duration estimation update
    input_text_single.change(
        update_duration_estimate,
        inputs=[input_text_single],
        outputs=[duration_estimate_display]
    )

    batch_file_input.upload(
        add_batch_prompts,
        inputs=[batch_file_input, batch_rows_state, next_batch_id_state, selected_entry],
        outputs=[batch_rows_state, next_batch_id_state, batch_file_input, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    load_dataset_button.click(
        load_dataset_entries,
        inputs=[dataset_path_input, batch_rows_state, next_batch_id_state, selected_entry],
        outputs=[batch_rows_state, next_batch_id_state, dataset_path_input, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    selected_entry.change(
        on_select_batch_entry,
        inputs=[selected_entry, batch_rows_state],
        outputs=[selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    batch_text_input.change(
        update_batch_text,
        inputs=[batch_text_input, batch_rows_state, selected_entry],
        outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    generate_all_button.click(
        generate_all_batch,
        inputs=[batch_rows_state, selected_entry, emo_control_method, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_sentence,
                *advanced_params],
        outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    regenerate_button.click(
        regenerate_batch_entry,
        inputs=[batch_rows_state, selected_entry, emo_control_method, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_sentence,
                *advanced_params],
        outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    delete_entry_button.click(
        delete_batch_entry,
        inputs=[batch_rows_state, selected_entry],
        outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    clear_entries_button.click(
        clear_batch_rows,
        inputs=[batch_rows_state, next_batch_id_state],
        outputs=[batch_rows_state, next_batch_id_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    # Model and GPU configuration handlers
    # Update metadata displays when model selection changes
    gpt_dropdown.change(
        handle_model_selection_change,
        inputs=[gpt_dropdown, gpt_paths_state],
        outputs=[tokenizer_info_display, vram_info_display]
    )

    load_models_button.click(
        handle_model_load,
        inputs=[gpt_dropdown, gpu_dropdown, gpt_paths_state],
        outputs=model_status
    )

    # Unload model button
    unload_model_button.click(
        unload_model_handler,
        inputs=[],
        outputs=[model_status, model_status_display, gpu_monitor_display]
    )

    # Refresh status button (replaces old refresh_monitor_button)
    refresh_status_button.click(
        refresh_status_handler,
        inputs=[],
        outputs=[model_status_display, gpu_monitor_display]
    )

    # History handlers
    refresh_history_button.click(
        refresh_history_gallery,
        inputs=[],
        outputs=[history_gallery, history_stats]
    )

    clear_history_button.click(
        clear_history_all,
        inputs=[],
        outputs=[history_gallery, history_stats]
    )

    # Compare Models tab handlers
    compare_model_a_dropdown.change(
        handle_compare_model_info_update,
        inputs=[compare_model_a_dropdown, gpt_paths_state],
        outputs=compare_model_a_info
    )

    compare_model_b_dropdown.change(
        handle_compare_model_info_update,
        inputs=[compare_model_b_dropdown, gpt_paths_state],
        outputs=compare_model_b_info
    )

    compare_generate_button.click(
        handle_compare_generate,
        inputs=[
            compare_model_a_dropdown,
            compare_model_b_dropdown,
            compare_text,
            compare_prompt_audio,
            compare_gpu_dropdown,
            gpt_paths_state
        ],
        outputs=[
            compare_status,
            compare_audio_a,
            compare_audio_b,
            compare_waveform_image,
            compare_metrics_display
        ]
    )

    # Training Setup tab handlers
    train_project_name.change(
        handle_train_project_name_change,
        inputs=[train_project_name],
        outputs=train_paths_display
    )

    train_tokenizer_button.click(
        handle_train_tokenizer,
        inputs=[
            train_project_name,
            train_vocab_size,
            train_char_coverage,
            train_tokenizer_manifest
        ],
        outputs=[train_tokenizer_status, train_tokenizer_output]
    )

    train_model_button.click(
        handle_train_model,
        inputs=[
            train_project_name,
            train_manifest,
            train_val_manifest,
            train_base_checkpoint,
            train_gpu_id,
            train_batch_size,
            train_epochs,
            train_learning_rate,
            train_warmup_steps,
            train_use_amp,
            gpt_paths_state
        ],
        outputs=[train_model_status, train_model_output]
    )

    train_refresh_checkpoints_button.click(
        handle_refresh_checkpoints,
        inputs=[train_project_name],
        outputs=[train_checkpoint_selector, train_checkpoint_info]
    )

    train_install_button.click(
        handle_install_model,
        inputs=[
            train_project_name,
            train_checkpoint_selector,
            train_description
        ],
        outputs=train_install_status
    )

    # Training Monitor event handlers
    monitor_refresh_projects.click(
        refresh_training_projects,
        inputs=[],
        outputs=monitor_project_selector
    )

    monitor_tb_start.click(
        start_tensorboard,
        inputs=[monitor_project_selector, monitor_tb_port],
        outputs=[monitor_tb_status, monitor_tb_frame]
    )

    monitor_tb_stop.click(
        stop_tensorboard,
        inputs=[],
        outputs=[monitor_tb_status, monitor_tb_frame]
    )

    monitor_refresh_status.click(
        refresh_training_status,
        inputs=[monitor_project_selector],
        outputs=monitor_status_display
    )

    monitor_refresh_plots.click(
        refresh_training_plots,
        inputs=[monitor_project_selector],
        outputs=[monitor_loss_plot, monitor_lr_plot]
    )

    # Auto-load projects on tab open
    monitor_project_selector.select(
        refresh_training_status,
        inputs=[monitor_project_selector],
        outputs=monitor_status_display
    )

    # Export handlers
    monitor_export_loss.click(
        lambda project: export_plot(project, "loss"),
        inputs=[monitor_project_selector],
        outputs=monitor_export_status
    )

    monitor_export_lr.click(
        lambda project: export_plot(project, "lr"),
        inputs=[monitor_project_selector],
        outputs=monitor_export_status
    )

    # Analysis handler
    monitor_analyze_button.click(
        analyze_training,
        inputs=[monitor_project_selector, monitor_target_step],
        outputs=monitor_alerts_display
    )

    # Comparison handlers
    monitor_refresh_projects.click(
        lambda: gr.update(choices=refresh_training_projects().choices),
        outputs=[monitor_compare_run1, monitor_compare_run2]
    )

    monitor_compare_runs_button.click(
        compare_training_runs,
        inputs=[monitor_compare_run1, monitor_compare_run2],
        outputs=[monitor_comparison_plot, monitor_comparison_stats]
    )



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
