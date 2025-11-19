"""
Training monitoring utilities for real-time visualization of training metrics.

This module provides tools to:
- Parse training logs from train_gpt_v2.py
- Extract metrics (loss, learning rate, etc.)
- Launch and manage TensorBoard instances
- Generate real-time plots
"""

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific step."""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    time_per_step: Optional[float] = None
    samples_per_sec: Optional[float] = None
    vram_gb: Optional[float] = None


class TrainingLogParser:
    """Parse training logs and extract metrics."""

    # Regex patterns for different log formats
    PATTERNS = {
        # Example: "Step 100/5000 | Epoch 1/10 | Loss: 2.345 | LR: 1.5e-4"
        'standard': re.compile(
            r'Step\s+(?P<step>\d+)/\d+\s*\|\s*'
            r'Epoch\s+(?P<epoch>\d+)/\d+\s*\|\s*'
            r'Loss:\s+(?P<loss>[\d.]+)\s*\|\s*'
            r'LR:\s+(?P<lr>[\d.e\-+]+)'
        ),
        # Example: "[2024-01-15 10:30:45] train_loss=2.345 lr=0.0001 step=100"
        'structured': re.compile(
            r'train_loss=(?P<loss>[\d.]+)\s+'
            r'lr=(?P<lr>[\d.e\-+]+)\s+'
            r'step=(?P<step>\d+)'
        ),
        # Grad norm: "grad_norm: 1.234"
        'grad_norm': re.compile(r'grad[_\s]norm[:\s=]+(?P<grad_norm>[\d.]+)'),
        # Time per step: "time: 0.123s"
        'time': re.compile(r'time[:\s=]+(?P<time>[\d.]+)'),
        # VRAM: "VRAM: 12.3 GB"
        'vram': re.compile(r'VRAM[:\s=]+(?P<vram>[\d.]+)\s*GB', re.IGNORECASE),
    }

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.metrics_history: List[TrainingMetrics] = []

    def parse_log_file(self) -> List[TrainingMetrics]:
        """Parse entire log file and return all metrics."""
        if not self.log_file.exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return []

        self.metrics_history = []

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    metric = self._parse_line(line)
                    if metric:
                        self.metrics_history.append(metric)
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")

        return self.metrics_history

    def parse_recent_lines(self, num_lines: int = 100) -> List[TrainingMetrics]:
        """Parse last N lines of log file."""
        if not self.log_file.exists():
            return []

        try:
            # Read last N lines efficiently
            with open(self.log_file, 'rb') as f:
                f.seek(0, 2)  # Go to end
                file_size = f.tell()

                # Estimate bytes to read (rough estimate)
                bytes_to_read = min(num_lines * 200, file_size)
                f.seek(max(0, file_size - bytes_to_read))

                lines = f.read().decode('utf-8', errors='ignore').split('\n')
                lines = lines[-num_lines:]

            recent_metrics = []
            for line in lines:
                metric = self._parse_line(line)
                if metric:
                    recent_metrics.append(metric)

            return recent_metrics
        except Exception as e:
            logger.error(f"Error parsing recent lines: {e}")
            return []

    def _parse_line(self, line: str) -> Optional[TrainingMetrics]:
        """Parse a single log line and extract metrics."""
        # Try standard format first
        match = self.PATTERNS['standard'].search(line)
        if match:
            return TrainingMetrics(
                step=int(match.group('step')),
                epoch=int(match.group('epoch')),
                loss=float(match.group('loss')),
                learning_rate=float(match.group('lr')),
                grad_norm=self._extract_optional(line, 'grad_norm'),
                time_per_step=self._extract_optional(line, 'time'),
                vram_gb=self._extract_optional(line, 'vram'),
            )

        # Try structured format
        match = self.PATTERNS['structured'].search(line)
        if match:
            return TrainingMetrics(
                step=int(match.group('step')),
                epoch=0,  # Not available in this format
                loss=float(match.group('loss')),
                learning_rate=float(match.group('lr')),
                grad_norm=self._extract_optional(line, 'grad_norm'),
                time_per_step=self._extract_optional(line, 'time'),
                vram_gb=self._extract_optional(line, 'vram'),
            )

        return None

    def _extract_optional(self, line: str, pattern_name: str) -> Optional[float]:
        """Extract optional metric from line."""
        if pattern_name in self.PATTERNS:
            match = self.PATTERNS[pattern_name].search(line)
            if match:
                return float(match.group(pattern_name.split('_')[-1]))
        return None

    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get the most recent metrics."""
        recent = self.parse_recent_lines(10)
        return recent[-1] if recent else None

    def get_metrics_for_plotting(self) -> Dict[str, List]:
        """Get metrics organized for plotting."""
        if not self.metrics_history:
            self.parse_log_file()

        return {
            'steps': [m.step for m in self.metrics_history],
            'losses': [m.loss for m in self.metrics_history],
            'learning_rates': [m.learning_rate for m in self.metrics_history],
            'grad_norms': [m.grad_norm for m in self.metrics_history if m.grad_norm],
            'vram_usage': [m.vram_gb for m in self.metrics_history if m.vram_gb],
        }


class TensorBoardManager:
    """Manage TensorBoard instances."""

    def __init__(self):
        self.tensorboard_process: Optional[subprocess.Popen] = None
        self.tensorboard_port: Optional[int] = None
        self.logdir: Optional[Path] = None

    def start(self, logdir: Path, port: int = 6006, host: str = "0.0.0.0") -> Tuple[bool, str]:
        """
        Start TensorBoard server.

        Args:
            logdir: Directory containing TensorBoard logs
            port: Port to run TensorBoard on
            host: Host to bind to (0.0.0.0 for all interfaces)

        Returns:
            (success, message/url)
        """
        if self.tensorboard_process and self.tensorboard_process.poll() is None:
            return False, f"TensorBoard already running on port {self.tensorboard_port}"

        if not logdir.exists():
            logdir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created logdir: {logdir}")

        try:
            # Kill any existing TensorBoard on this port
            try:
                subprocess.run(
                    ["pkill", "-f", f"tensorboard.*--port.*{port}"],
                    stderr=subprocess.DEVNULL,
                    timeout=5
                )
                time.sleep(1)
            except:
                pass

            # Start TensorBoard
            cmd = [
                "tensorboard",
                "--logdir", str(logdir),
                "--port", str(port),
                "--host", host,
                "--reload_interval", "10",  # Refresh every 10 seconds
            ]

            logger.info(f"Starting TensorBoard: {' '.join(cmd)}")

            self.tensorboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait a bit and check if it started successfully
            time.sleep(2)

            if self.tensorboard_process.poll() is not None:
                # Process ended immediately - error
                _, stderr = self.tensorboard_process.communicate(timeout=1)
                return False, f"TensorBoard failed to start: {stderr}"

            self.tensorboard_port = port
            self.logdir = logdir

            url = f"http://localhost:{port}"
            logger.info(f"TensorBoard started at {url}")

            return True, url

        except FileNotFoundError:
            return False, "TensorBoard not installed. Install with: pip install tensorboard"
        except Exception as e:
            logger.error(f"Error starting TensorBoard: {e}")
            return False, f"Failed to start TensorBoard: {e}"

    def stop(self) -> Tuple[bool, str]:
        """Stop TensorBoard server."""
        if not self.tensorboard_process or self.tensorboard_process.poll() is not None:
            return False, "TensorBoard is not running"

        try:
            self.tensorboard_process.terminate()
            try:
                self.tensorboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tensorboard_process.kill()
                self.tensorboard_process.wait(timeout=2)

            logger.info("TensorBoard stopped")
            self.tensorboard_process = None
            self.tensorboard_port = None

            return True, "TensorBoard stopped successfully"
        except Exception as e:
            logger.error(f"Error stopping TensorBoard: {e}")
            return False, f"Failed to stop TensorBoard: {e}"

    def is_running(self) -> bool:
        """Check if TensorBoard is running."""
        return self.tensorboard_process is not None and self.tensorboard_process.poll() is None

    def get_url(self) -> Optional[str]:
        """Get TensorBoard URL if running."""
        if self.is_running() and self.tensorboard_port:
            return f"http://localhost:{self.tensorboard_port}"
        return None


def find_training_logs(project_name: str, training_root: Path = Path("training")) -> List[Path]:
    """
    Find training log files for a project.

    Args:
        project_name: Project name (e.g., 'catalan')
        training_root: Root training directory

    Returns:
        List of log file paths
    """
    project_dir = training_root / project_name / "checkpoints"

    if not project_dir.exists():
        return []

    # Look for common log file patterns
    log_patterns = ["training.log", "train.log", "*.log"]
    log_files = []

    for pattern in log_patterns:
        log_files.extend(project_dir.glob(pattern))

    # Also check runs/ subdirectory (TensorBoard events)
    runs_dir = project_dir / "runs"
    if runs_dir.exists():
        log_files.extend(runs_dir.glob("**/*.log"))

    return sorted(set(log_files))


def get_tensorboard_logdir(project_name: str, training_root: Path = Path("training")) -> Path:
    """Get TensorBoard log directory for a project."""
    return training_root / project_name / "checkpoints" / "runs"
