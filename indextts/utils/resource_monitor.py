"""
GPU resource monitoring utilities for IndexTTS.
Provides real-time VRAM usage, temperature, and utilization tracking.
"""
import time
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from threading import Lock

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GPUStats:
    """Statistics for a single GPU."""
    device_id: int
    name: str
    temperature: Optional[int]  # Celsius
    memory_used: int  # MB
    memory_total: int  # MB
    memory_percent: float
    utilization: Optional[int]  # Percent
    power_draw: Optional[float]  # Watts
    power_limit: Optional[float]  # Watts


class ResourceMonitor:
    """
    Monitor GPU resources with caching to avoid overhead.

    Thread-safe singleton that caches nvidia-smi queries.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cache: Dict[int, GPUStats] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 2.0  # Cache for 2 seconds
        self._nvidia_smi_available = self._check_nvidia_smi()

        # Build mapping from PyTorch device IDs to nvidia-smi indexes
        # This is needed because PyTorch and nvidia-smi may enumerate GPUs in different orders
        self._pytorch_to_nvidiasmi_map: Dict[int, int] = self._build_device_mapping()

        self._initialized = True

    @staticmethod
    def _check_nvidia_smi() -> bool:
        """Check if nvidia-smi is available."""
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _build_device_mapping(self) -> Dict[int, int]:
        """
        Build mapping from PyTorch device IDs to nvidia-smi indexes using GPU UUIDs.

        PyTorch and nvidia-smi may enumerate GPUs in different orders.
        We use GPU UUIDs as a stable identifier to create the correct mapping.

        Returns:
            Dictionary mapping PyTorch device ID to nvidia-smi index
        """
        mapping = {}

        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return mapping

        if not self._nvidia_smi_available:
            # If nvidia-smi isn't available, assume 1:1 mapping
            for i in range(torch.cuda.device_count()):
                mapping[i] = i
            return mapping

        try:
            # Get UUID mapping from nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                # Fallback to 1:1 mapping
                for i in range(torch.cuda.device_count()):
                    mapping[i] = i
                return mapping

            # Parse nvidia-smi output to build UUID -> smi_index mapping
            nvidiasmi_uuid_to_index = {}
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    smi_index = int(parts[0])
                    uuid = parts[1]
                    nvidiasmi_uuid_to_index[uuid] = smi_index

            # Get UUID for each PyTorch device using nvidia-smi -L
            result_list = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result_list.returncode != 0:
                # Fallback to 1:1 mapping
                for i in range(torch.cuda.device_count()):
                    mapping[i] = i
                return mapping

            # Match PyTorch device names with nvidia-smi -L output to get UUIDs
            import re
            for pytorch_id in range(torch.cuda.device_count()):
                pytorch_name = torch.cuda.get_device_name(pytorch_id)

                # Find this GPU in nvidia-smi -L output
                for line in result_list.stdout.strip().split('\n'):
                    if pytorch_name in line:
                        # Extract UUID from line like: "GPU 0: ... (UUID: GPU-xxx)"
                        match = re.search(r'UUID: (GPU-[\w-]+)', line)
                        if match:
                            uuid = match.group(1)
                            if uuid in nvidiasmi_uuid_to_index:
                                mapping[pytorch_id] = nvidiasmi_uuid_to_index[uuid]
                                break

            # If we didn't get a complete mapping, fallback to 1:1
            if len(mapping) != torch.cuda.device_count():
                mapping.clear()
                for i in range(torch.cuda.device_count()):
                    mapping[i] = i

        except Exception:
            # On any error, use 1:1 mapping as fallback
            for i in range(torch.cuda.device_count()):
                mapping[i] = i

        return mapping

    def get_gpu_stats(self, device_id: int = 0) -> Optional[GPUStats]:
        """
        Get statistics for a specific GPU.

        Args:
            device_id: GPU device ID (0-based)

        Returns:
            GPUStats object or None if unavailable
        """
        # Check cache first
        current_time = time.time()
        if current_time - self._cache_timestamp < self._cache_ttl:
            return self._cache.get(device_id)

        # Refresh cache
        self._refresh_cache()
        return self._cache.get(device_id)

    def get_all_gpu_stats(self) -> List[GPUStats]:
        """
        Get statistics for all available GPUs.

        Returns:
            List of GPUStats objects
        """
        current_time = time.time()
        if current_time - self._cache_timestamp < self._cache_ttl:
            return list(self._cache.values())

        self._refresh_cache()
        return list(self._cache.values())

    def _refresh_cache(self):
        """Refresh the statistics cache."""
        self._cache.clear()

        if not self._nvidia_smi_available:
            # Fall back to PyTorch-only stats
            self._refresh_pytorch_only()
            return

        try:
            # Query nvidia-smi for detailed stats
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                self._parse_nvidia_smi_output(result.stdout)
            else:
                self._refresh_pytorch_only()

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._refresh_pytorch_only()

        self._cache_timestamp = time.time()

    def _parse_nvidia_smi_output(self, output: str):
        """Parse nvidia-smi CSV output."""
        # Invert the mapping: nvidia-smi index -> PyTorch device ID
        nvidiasmi_to_pytorch = {smi_idx: pytorch_id for pytorch_id, smi_idx in self._pytorch_to_nvidiasmi_map.items()}

        for line in output.strip().split('\n'):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 8:
                continue

            try:
                nvidiasmi_index = int(parts[0])  # This is nvidia-smi's index
                name = parts[1]
                temp = int(parts[2]) if parts[2] and parts[2] != 'N/A' else None
                mem_used = int(float(parts[3]))  # Handle decimal values
                mem_total = int(float(parts[4]))
                utilization = int(parts[5]) if parts[5] and parts[5] != 'N/A' else None
                power_draw = float(parts[6]) if parts[6] and parts[6] != 'N/A' else None
                power_limit = float(parts[7]) if parts[7] and parts[7] != 'N/A' else None

                mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0.0

                # Convert nvidia-smi index to PyTorch device ID
                pytorch_device_id = nvidiasmi_to_pytorch.get(nvidiasmi_index, nvidiasmi_index)

                # Store stats using PyTorch device ID as the key
                self._cache[pytorch_device_id] = GPUStats(
                    device_id=pytorch_device_id,  # Use PyTorch device ID
                    name=name,
                    temperature=temp,
                    memory_used=mem_used,
                    memory_total=mem_total,
                    memory_percent=mem_percent,
                    utilization=utilization,
                    power_draw=power_draw,
                    power_limit=power_limit
                )
            except (ValueError, IndexError):
                continue

    def _refresh_pytorch_only(self):
        """Refresh cache using PyTorch only (limited info)."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return

        for device_id in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(device_id)
                mem_allocated = torch.cuda.memory_allocated(device_id) // (1024 ** 2)  # Convert to MB
                mem_reserved = torch.cuda.memory_reserved(device_id) // (1024 ** 2)
                mem_total = props.total_memory // (1024 ** 2)

                # Use reserved memory as a more accurate measure of usage
                mem_used = mem_reserved
                mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0.0

                self._cache[device_id] = GPUStats(
                    device_id=device_id,
                    name=props.name,
                    temperature=None,  # Not available via PyTorch
                    memory_used=mem_used,
                    memory_total=mem_total,
                    memory_percent=mem_percent,
                    utilization=None,  # Not available via PyTorch
                    power_draw=None,
                    power_limit=None
                )
            except Exception:
                continue

    def format_gpu_stats(self, stats: GPUStats, compact: bool = False) -> str:
        """
        Format GPU statistics as a human-readable string.

        Args:
            stats: GPUStats object
            compact: If True, return single-line format

        Returns:
            Formatted string
        """
        if compact:
            temp_str = f"{stats.temperature}°C" if stats.temperature is not None else "N/A"
            util_str = f"{stats.utilization}%" if stats.utilization is not None else "N/A"
            return (
                f"GPU {stats.device_id}: {stats.name} | "
                f"VRAM: {stats.memory_used}MB/{stats.memory_total}MB ({stats.memory_percent:.1f}%) | "
                f"Temp: {temp_str} | Util: {util_str}"
            )
        else:
            lines = [
                f"GPU {stats.device_id}: {stats.name}",
                f"  VRAM: {stats.memory_used}MB / {stats.memory_total}MB ({stats.memory_percent:.1f}%)",
            ]

            if stats.temperature is not None:
                lines.append(f"  Temperature: {stats.temperature}°C")

            if stats.utilization is not None:
                lines.append(f"  Utilization: {stats.utilization}%")

            if stats.power_draw is not None and stats.power_limit is not None:
                lines.append(f"  Power: {stats.power_draw:.1f}W / {stats.power_limit:.1f}W")

            return "\n".join(lines)

    def get_vram_summary(self, device_id: int = 0) -> Tuple[int, int, float]:
        """
        Get VRAM usage summary for a specific GPU.

        Args:
            device_id: GPU device ID

        Returns:
            Tuple of (used_mb, total_mb, percent)
        """
        stats = self.get_gpu_stats(device_id)
        if stats:
            return stats.memory_used, stats.memory_total, stats.memory_percent
        return 0, 0, 0.0

    def estimate_available_vram(self, device_id: int = 0) -> int:
        """
        Estimate available VRAM in MB.

        Args:
            device_id: GPU device ID

        Returns:
            Available VRAM in MB (conservative estimate)
        """
        stats = self.get_gpu_stats(device_id)
        if stats:
            # Leave 500MB buffer for system and overhead
            available = stats.memory_total - stats.memory_used - 500
            return max(0, available)
        return 0

    def can_fit_model(self, model_size_mb: int, device_id: int = 0) -> bool:
        """
        Check if a model of given size can fit in GPU memory.

        Args:
            model_size_mb: Estimated model size in MB
            device_id: GPU device ID

        Returns:
            True if model can fit
        """
        available = self.estimate_available_vram(device_id)
        return available >= model_size_mb

    def predict_oom_risk(self, device_id: int = 0) -> str:
        """
        Predict Out-Of-Memory risk level.

        Args:
            device_id: GPU device ID

        Returns:
            Risk level: "low", "medium", "high", or "critical"
        """
        stats = self.get_gpu_stats(device_id)
        if not stats:
            return "unknown"

        percent = stats.memory_percent

        if percent < 70:
            return "low"
        elif percent < 85:
            return "medium"
        elif percent < 95:
            return "high"
        else:
            return "critical"


# Global singleton instance
_monitor = ResourceMonitor()


def get_monitor() -> ResourceMonitor:
    """Get the global ResourceMonitor instance."""
    return _monitor


# Convenience functions
def get_gpu_stats(device_id: int = 0) -> Optional[GPUStats]:
    """Get statistics for a specific GPU."""
    return _monitor.get_gpu_stats(device_id)


def get_all_gpu_stats() -> List[GPUStats]:
    """Get statistics for all available GPUs."""
    return _monitor.get_all_gpu_stats()


def get_vram_summary(device_id: int = 0) -> Tuple[int, int, float]:
    """Get VRAM usage summary (used_mb, total_mb, percent)."""
    return _monitor.get_vram_summary(device_id)


def format_vram_bar(percent: float, width: int = 20) -> str:
    """
    Create a text-based progress bar for VRAM usage.

    Args:
        percent: Usage percentage (0-100)
        width: Width of the bar in characters

    Returns:
        ASCII progress bar string
    """
    filled = int(width * percent / 100)
    empty = width - filled

    # Color coding (for terminals that support it)
    if percent < 70:
        color = "🟢"  # Green
    elif percent < 85:
        color = "🟡"  # Yellow
    else:
        color = "🔴"  # Red

    bar = "█" * filled + "░" * empty
    return f"{color} [{bar}] {percent:.1f}%"
