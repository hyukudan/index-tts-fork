"""
Audio generation history manager for IndexTTS.
Tracks generated audio files with metadata for gallery display.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid


@dataclass
class GenerationRecord:
    """Record of a single audio generation."""
    id: str
    timestamp: str
    text: str
    audio_path: str
    model_gpt: str
    model_tokenizer: str
    prompt_audio: Optional[str] = None
    emotion_mode: Optional[str] = None
    emotion_audio: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    max_mel_tokens: int = 1500
    duration_seconds: Optional[float] = None
    rtf: Optional[float] = None  # Real-time factor
    seed: Optional[int] = None


class AudioHistoryManager:
    """
    Manage audio generation history with persistent storage.

    Stores generated audio files and metadata in session-based directories.
    """

    def __init__(self, base_dir: str = "outputs/history"):
        """
        Initialize the history manager.

        Args:
            base_dir: Base directory for history storage
        """
        self.base_dir = Path(base_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.session_dir / "metadata.jsonl"
        self.records: Dict[str, GenerationRecord] = {}

        # Load existing records if resuming session
        self._load_records()

    def _load_records(self):
        """Load existing records from metadata file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record_dict = json.loads(line)
                        record = GenerationRecord(**record_dict)
                        self.records[record.id] = record
            except Exception as e:
                print(f"Warning: Could not load history metadata: {e}")

    def _save_record(self, record: GenerationRecord):
        """Append a record to the metadata file."""
        try:
            with open(self.metadata_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(record), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Could not save record: {e}")

    def add_generation(
        self,
        audio_path: str,
        text: str,
        model_gpt: str,
        model_tokenizer: str,
        **kwargs
    ) -> GenerationRecord:
        """
        Add a new generation to the history.

        Args:
            audio_path: Path to generated audio file
            text: Input text
            model_gpt: GPT checkpoint name
            model_tokenizer: Tokenizer name
            **kwargs: Additional metadata (prompt_audio, emotion_mode, etc.)

        Returns:
            GenerationRecord object
        """
        # Generate unique ID
        record_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Copy audio to history directory
        audio_src = Path(audio_path)
        audio_dest = self.session_dir / f"{record_id}_{audio_src.name}"

        try:
            shutil.copy2(audio_src, audio_dest)
        except Exception as e:
            print(f"Warning: Could not copy audio to history: {e}")
            audio_dest = Path(audio_path)  # Use original path as fallback

        # Create record
        record = GenerationRecord(
            id=record_id,
            timestamp=timestamp,
            text=text,
            audio_path=str(audio_dest),
            model_gpt=model_gpt,
            model_tokenizer=model_tokenizer,
            **kwargs
        )

        # Save to memory and disk
        self.records[record_id] = record
        self._save_record(record)

        return record

    def get_record(self, record_id: str) -> Optional[GenerationRecord]:
        """Get a specific record by ID."""
        return self.records.get(record_id)

    def get_all_records(self, limit: Optional[int] = None, reverse: bool = True) -> List[GenerationRecord]:
        """
        Get all records, optionally limited and sorted.

        Args:
            limit: Maximum number of records to return
            reverse: If True, return newest first

        Returns:
            List of GenerationRecord objects
        """
        records = list(self.records.values())

        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp, reverse=reverse)

        if limit:
            records = records[:limit]

        return records

    def delete_record(self, record_id: str) -> bool:
        """
        Delete a record and its audio file.

        Args:
            record_id: Record ID to delete

        Returns:
            True if deleted successfully
        """
        record = self.records.get(record_id)
        if not record:
            return False

        # Delete audio file
        try:
            audio_path = Path(record.audio_path)
            if audio_path.exists():
                audio_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete audio file: {e}")

        # Remove from memory
        del self.records[record_id]

        # Rewrite metadata file
        self._rewrite_metadata()

        return True

    def _rewrite_metadata(self):
        """Rewrite the metadata file with current records."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                for record in self.records.values():
                    json.dump(asdict(record), f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            print(f"Warning: Could not rewrite metadata: {e}")

    def clear_history(self):
        """Clear all history records and files."""
        # Delete all audio files
        for record in self.records.values():
            try:
                audio_path = Path(record.audio_path)
                if audio_path.exists() and audio_path.parent == self.session_dir:
                    audio_path.unlink()
            except Exception:
                pass

        # Clear memory and metadata
        self.records.clear()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

    def get_statistics(self) -> Dict:
        """Get statistics about the current history."""
        if not self.records:
            return {
                "total_generations": 0,
                "total_duration": 0.0,
                "avg_rtf": 0.0,
            }

        total = len(self.records)
        total_duration = sum(r.duration_seconds or 0.0 for r in self.records.values())
        rtf_values = [r.rtf for r in self.records.values() if r.rtf is not None]
        avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0

        return {
            "total_generations": total,
            "total_duration": total_duration,
            "avg_rtf": avg_rtf,
        }

    def format_record_summary(self, record: GenerationRecord, max_text_len: int = 50) -> str:
        """
        Format a record as a summary string.

        Args:
            record: GenerationRecord object
            max_text_len: Maximum text length to display

        Returns:
            Formatted summary string
        """
        # Truncate text if needed
        text = record.text
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."

        # Format timestamp
        try:
            dt = datetime.fromisoformat(record.timestamp)
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = "Unknown"

        # Format model names (show filename only)
        gpt_name = Path(record.model_gpt).stem if record.model_gpt else "Unknown"
        tok_name = Path(record.model_tokenizer).stem if record.model_tokenizer else "Unknown"

        parts = [
            f"[{time_str}]",
            f"\"{text}\"",
            f"(Model: {gpt_name}, Tokenizer: {tok_name})"
        ]

        if record.rtf is not None:
            parts.append(f"RTF: {record.rtf:.3f}")

        return " ".join(parts)

    def export_to_json(self, output_path: str):
        """
        Export all records to a JSON file.

        Args:
            output_path: Path to output JSON file
        """
        records_list = [asdict(r) for r in self.records.values()]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records_list, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_from_session(session_dir: str) -> "AudioHistoryManager":
        """
        Load an existing session.

        Args:
            session_dir: Path to session directory

        Returns:
            AudioHistoryManager instance
        """
        manager = AudioHistoryManager.__new__(AudioHistoryManager)
        manager.base_dir = Path(session_dir).parent
        manager.session_dir = Path(session_dir)
        manager.session_id = manager.session_dir.name
        manager.metadata_file = manager.session_dir / "metadata.jsonl"
        manager.records = {}
        manager._load_records()

        return manager

    @staticmethod
    def list_sessions(base_dir: str = "outputs/history") -> List[str]:
        """
        List all available sessions.

        Args:
            base_dir: Base history directory

        Returns:
            List of session directory names (sorted newest first)
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        sessions = [d.name for d in base_path.iterdir() if d.is_dir()]
        sessions.sort(reverse=True)

        return sessions

    def get_gallery_data(self, limit: int = 50) -> List[tuple]:
        """
        Get data formatted for Gradio Gallery component.

        Args:
            limit: Maximum number of items

        Returns:
            List of tuples (audio_path, caption)
        """
        records = self.get_all_records(limit=limit, reverse=True)
        gallery_data = []

        for record in records:
            caption = self.format_record_summary(record)
            gallery_data.append((record.audio_path, caption))

        return gallery_data


# Global instance for the current session
_history_manager = None


def get_history_manager(base_dir: str = "outputs/history") -> AudioHistoryManager:
    """Get the global AudioHistoryManager instance."""
    global _history_manager
    if _history_manager is None:
        _history_manager = AudioHistoryManager(base_dir)
    return _history_manager


def reset_history_manager():
    """Reset the global history manager (for testing)."""
    global _history_manager
    _history_manager = None
