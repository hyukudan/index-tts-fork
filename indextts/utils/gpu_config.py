"""
GPU configuration manager for IndexTTS.
Handles multi-GPU detection, selection, and persistent configuration.
"""
import os
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GPUConfig:
    """Manage GPU configuration and selection."""

    CONFIG_FILE = Path.home() / ".indextts" / "gpu_config.json"

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load saved GPU configuration."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_config(self):
        """Save GPU configuration."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)

    @staticmethod
    def detect_platform() -> Dict[str, any]:
        """Detect platform details including WSL."""
        import torch

        info = {
            "system": platform.system(),
            "is_wsl": False,
            "wsl_version": None,
            "is_linux": platform.system() == "Linux",
            "is_windows": platform.system() == "Windows",
            "is_darwin": platform.system() == "Darwin",
            "cuda_available": torch.cuda.is_available(),
        }

        # Detect WSL and version
        if info["is_linux"]:
            try:
                with open('/proc/version', 'r') as f:
                    content = f.read().lower()
                    if 'microsoft' in content or 'wsl' in content:
                        info["is_wsl"] = True
                        # WSL2 uses different kernel
                        if 'wsl2' in content or 'microsoft-standard' in content:
                            info["wsl_version"] = "WSL2"
                        else:
                            info["wsl_version"] = "WSL1"
            except:
                pass

        return info

    @staticmethod
    def check_wsl_gpu_support() -> Dict[str, any]:
        """Check if WSL has proper GPU support."""
        result = {
            "supported": False,
            "nvidia_smi_works": False,
            "cuda_available": False,
            "driver_version": None,
            "issues": [],
        }

        # Check if nvidia-smi works
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                timeout=5
            ).decode().strip()
            result["nvidia_smi_works"] = True
            result["driver_version"] = output
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            result["issues"].append("nvidia-smi not found or not working")

        # Check CUDA availability
        try:
            import torch
            result["cuda_available"] = torch.cuda.is_available()
            if not result["cuda_available"]:
                result["issues"].append("CUDA not available in PyTorch")
        except:
            result["issues"].append("PyTorch not installed or import failed")

        result["supported"] = result["nvidia_smi_works"] and result["cuda_available"]
        return result

    @staticmethod
    def get_gpu_info() -> List[Dict[str, any]]:
        """Get information about all available GPUs."""
        import torch

        if not torch.cuda.is_available():
            return []

        gpus = []
        device_count = torch.cuda.device_count()

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            compute_cap = torch.cuda.get_device_capability(i)

            # Classify architecture
            arch_name = "Unknown"
            if compute_cap[0] >= 10:
                arch_name = "Blackwell"
            elif compute_cap[0] == 9:
                arch_name = "Hopper"
            elif compute_cap[0] == 8:
                if compute_cap[1] >= 9:
                    arch_name = "Ada Lovelace"
                elif compute_cap[1] >= 6:
                    arch_name = "Ampere"
            elif compute_cap[0] == 7:
                if compute_cap[1] >= 5:
                    arch_name = "Turing"
                else:
                    arch_name = "Volta"

            gpu_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
                "architecture": arch_name,
                "multi_processor_count": props.multi_processor_count,
                "is_blackwell": compute_cap[0] >= 10,
                "is_ada_or_newer": compute_cap[0] >= 8 and compute_cap[1] >= 9,
                "supports_bf16": compute_cap[0] >= 8,  # Ampere and newer
                "suggested_workers": max(1, min(int(props.total_memory / 8e9), 8)),
            }

            gpus.append(gpu_info)

        return gpus

    @staticmethod
    def check_flash_attention() -> Dict[str, any]:
        """Check Flash Attention availability and compatibility."""
        result = {
            "installed": False,
            "version": None,
            "compatible": False,
            "reason": None,
        }

        try:
            import flash_attn
            result["installed"] = True
            result["version"] = flash_attn.__version__
            result["compatible"] = True
        except ImportError:
            result["reason"] = "Not installed"
        except Exception as e:
            result["installed"] = True
            result["reason"] = f"Import error: {e}"

        return result

    def get_selected_gpu(self, cmd_arg: Optional[int] = None) -> Optional[int]:
        """
        Get selected GPU ID.
        Priority: cmd argument > saved config > None (will prompt)
        """
        if cmd_arg is not None:
            return cmd_arg

        return self.config.get("selected_gpu_id")

    def interactive_gpu_selection(self, gpus: List[Dict], platform_info: Dict) -> int:
        """Interactive GPU selection for first-time setup."""
        print("\n" + "="*70)
        print("🚀 IndexTTS GPU Configuration")
        print("="*70)

        # Platform info with WSL-specific warnings
        if platform_info["is_wsl"]:
            wsl_version = platform_info.get("wsl_version", "Unknown")
            print(f"\n📍 Platform: {wsl_version}")

            if wsl_version == "WSL1":
                print("   ❌ WARNING: WSL1 does NOT support GPU!")
                print("   💡 Please upgrade to WSL2:")
                print("      wsl --set-version <distro-name> 2")
                print("      See: https://docs.microsoft.com/en-us/windows/wsl/install")

                # Check if GPU is actually working despite WSL1
                if not platform_info.get("cuda_available"):
                    print("\n❌ GPU not available. Exiting...")
                    exit(1)
            else:
                print("   ✅ WSL2 detected - GPU support available")

                # Check WSL GPU support
                wsl_gpu = self.check_wsl_gpu_support()
                if not wsl_gpu["supported"]:
                    print(f"\n⚠️  WSL GPU Support Issues:")
                    for issue in wsl_gpu["issues"]:
                        print(f"   • {issue}")
                    print("\n💡 To fix WSL GPU support:")
                    print("   1. Update Windows to latest version")
                    print("   2. Install NVIDIA drivers for WSL:")
                    print("      https://docs.nvidia.com/cuda/wsl-user-guide/index.html")
                    print("   3. Restart WSL: wsl --shutdown")
                else:
                    print(f"   ✅ NVIDIA Driver: {wsl_gpu['driver_version']}")
                    print("   💡 WSL2 often provides better performance than Windows native")

        elif platform_info["is_windows"]:
            print(f"\n📍 Platform: Windows (Native)")
            print("   💡 Consider using WSL2 for potentially better performance")
            print("   💡 Install WSL2: wsl --install")
        elif platform_info["is_linux"]:
            print(f"\n📍 Platform: Linux (Native)")
        elif platform_info["is_darwin"]:
            print(f"\n📍 Platform: macOS (GPU features limited)")

        # GPU list
        print(f"\n🎮 Detected {len(gpus)} GPU(s):")
        print()

        for gpu in gpus:
            print(f"  [{gpu['id']}] {gpu['name']}")
            print(f"      Architecture: {gpu['architecture']} (sm_{gpu['compute_capability']})")
            print(f"      Memory: {gpu['total_memory_gb']:.1f} GB")
            print(f"      Suggested workers: {gpu['suggested_workers']}")

            # Special notes
            if gpu['is_blackwell']:
                print(f"      💎 Blackwell GPU detected!")
                print(f"         • BF16 recommended for stability")
                print(f"         • Flash Attention: build from source required")
                if platform_info["is_wsl"]:
                    print(f"         • WSL: Ensure latest NVIDIA drivers (560+)")
            elif gpu['is_ada_or_newer']:
                print(f"      ✨ Ada Lovelace GPU - excellent performance")
                print(f"         • Flash Attention available via pip")

            print()

        # Flash Attention status
        flash_info = self.check_flash_attention()
        if flash_info["installed"]:
            print(f"⚡ Flash Attention: Installed (v{flash_info['version']})")
        else:
            print(f"⚠️  Flash Attention: Not installed")
            print(f"   Install with: uv sync --extra flashattn")
            if any(gpu['is_blackwell'] for gpu in gpus):
                print(f"   ⚠️  Blackwell detected: Build from source required!")
                print(f"      See INSTALLATION_UPDATED.md for instructions")
            if platform_info["is_wsl"]:
                print(f"   💡 In WSL, build from source recommended for best compatibility")

        print()
        print("="*70)

        # Selection
        if len(gpus) == 1:
            print(f"\n✅ Only one GPU detected. Using GPU 0: {gpus[0]['name']}")
            selected = 0
        else:
            while True:
                try:
                    choice = input(f"\n🎯 Select GPU to use [0-{len(gpus)-1}] (or 'q' to quit): ").strip()
                    if choice.lower() == 'q':
                        print("Exiting...")
                        exit(0)

                    selected = int(choice)
                    if 0 <= selected < len(gpus):
                        break
                    else:
                        print(f"❌ Invalid choice. Please select 0-{len(gpus)-1}")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    exit(0)

        # Save configuration
        self.config["selected_gpu_id"] = selected
        self.config["platform_info"] = platform_info
        self.config["last_detected_gpus"] = gpus
        self._save_config()

        print(f"\n✅ Configuration saved to: {self.CONFIG_FILE}")
        print(f"   Selected GPU: {gpus[selected]['name']}")
        print(f"\n💡 To change GPU later:")
        print(f"   • Use --gpu <id> argument")
        print(f"   • Or delete: {self.CONFIG_FILE}")

        if platform_info["is_wsl"]:
            print(f"\n💡 WSL Tips:")
            print(f"   • Use Linux paths for best compatibility")
            print(f"   • Access Windows files via /mnt/c/...")
            print(f"   • If you see multiprocessing errors, try fewer workers")

        print()

        return selected

    def ensure_gpu_selected(self, cmd_arg: Optional[int] = None) -> Optional[int]:
        """
        Ensure a GPU is selected. Will prompt if needed.
        Returns None if no CUDA available.
        """
        import torch

        if not torch.cuda.is_available():
            return None

        gpus = self.get_gpu_info()
        if not gpus:
            return None

        # Check command line argument
        if cmd_arg is not None:
            if 0 <= cmd_arg < len(gpus):
                print(f"🎯 Using GPU {cmd_arg} (from --gpu argument): {gpus[cmd_arg]['name']}")
                return cmd_arg
            else:
                print(f"❌ Invalid GPU ID: {cmd_arg}. Available: 0-{len(gpus)-1}")
                exit(1)

        # Check saved config
        saved_gpu = self.config.get("selected_gpu_id")
        if saved_gpu is not None and 0 <= saved_gpu < len(gpus):
            return saved_gpu

        # Need interactive selection
        platform_info = self.detect_platform()
        return self.interactive_gpu_selection(gpus, platform_info)

    def print_config_summary(self, gpu_id: Optional[int] = None):
        """Print current configuration summary."""
        import torch

        if gpu_id is None or not torch.cuda.is_available():
            print("⚠️  No GPU selected or CUDA not available")
            return

        gpus = self.get_gpu_info()
        if gpu_id >= len(gpus):
            print(f"❌ GPU {gpu_id} not found")
            return

        gpu = gpus[gpu_id]
        platform_info = self.detect_platform()
        flash_info = self.check_flash_attention()

        print("\n" + "="*70)
        print("📊 IndexTTS Configuration Summary")
        print("="*70)

        # Platform
        if platform_info["is_wsl"]:
            print(f"Platform: WSL2 on Windows")
        elif platform_info["is_windows"]:
            print(f"Platform: Windows")
        elif platform_info["is_linux"]:
            print(f"Platform: Linux")

        # GPU
        print(f"\nGPU {gpu_id}: {gpu['name']}")
        print(f"  • Architecture: {gpu['architecture']} (sm_{gpu['compute_capability']})")
        print(f"  • Memory: {gpu['total_memory_gb']:.1f} GB")
        print(f"  • CUDA: {torch.version.cuda}")
        print(f"  • PyTorch: {torch.__version__}")

        # Flash Attention
        if flash_info["installed"]:
            print(f"\n⚡ Flash Attention: v{flash_info['version']}")
        else:
            print(f"\n⚠️  Flash Attention: Not installed ({flash_info['reason']})")

        # Recommendations
        print(f"\n💡 Recommendations:")
        print(f"  • Suggested parallel workers: {gpu['suggested_workers']}")

        if gpu['is_blackwell']:
            print(f"  • Use BF16 for better stability (FP16 may cause NaN)")
            if not flash_info["installed"]:
                print(f"  • Build Flash Attention from source for best performance")

        if platform_info["is_windows"] and not platform_info["is_wsl"]:
            print(f"  • Consider using WSL2 for better performance")

        print("="*70 + "\n")


def setup_gpu(cmd_gpu_id: Optional[int] = None) -> Tuple[Optional[int], Dict]:
    """
    Setup GPU configuration. Call this at application startup.

    Args:
        cmd_gpu_id: GPU ID from command line argument (optional)

    Returns:
        (gpu_id, gpu_info_dict) or (None, {}) if no CUDA
    """
    import torch

    config = GPUConfig()

    # Ensure GPU is selected (will prompt if needed)
    gpu_id = config.ensure_gpu_selected(cmd_gpu_id)

    if gpu_id is None:
        return None, {}

    # Set PyTorch to use selected GPU
    torch.cuda.set_device(gpu_id)

    # Get GPU info
    gpus = config.get_gpu_info()
    gpu_info = gpus[gpu_id] if gpu_id < len(gpus) else {}

    # Print summary
    config.print_config_summary(gpu_id)

    return gpu_id, gpu_info
