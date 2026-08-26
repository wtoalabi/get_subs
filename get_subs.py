#!/usr/bin/env python3
"""Recursively create real-time SRT subtitles with Qwen3-ASR and its aligner.

The command intentionally keeps orchestration in the Python standard library.
Heavy runtime imports occur only after argument validation and explicit model
provisioning, which makes startup failures direct and leaves partial work
resumable through a small sidecar checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence


APP_NAME = "get_subs"
APP_VERSION = "1.0.0"
ASR_REPOSITORY = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_REPOSITORY = "Qwen/Qwen3-ForcedAligner-0.6B"
CHECKPOINT_VERSION = 1
PASS_COUNT = 6
DEFAULT_MODEL_CACHE = Path(
    os.environ.get("GET_SUBS_MODEL_CACHE", "~/.cache/get_subs/huggingface")
).expanduser()
VIDEO_EXTENSIONS = frozenset(
    {
        ".3g2",
        ".3gp",
        ".asf",
        ".avi",
        ".divx",
        ".f4v",
        ".flv",
        ".m2ts",
        ".m2v",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".mxf",
        ".ogv",
        ".rm",
        ".rmvb",
        ".ts",
        ".vob",
        ".webm",
        ".wmv",
    }
)
STRONG_PUNCTUATION = frozenset(".!?。！？")


@dataclasses.dataclass(frozen=True)
class VideoInfo:
    """Describe a discovered video and the audio metadata needed for planning."""

    path: Path
    output_path: Path
    duration: float
    audio_codec: str
    sample_rate: Optional[int]
    channels: Optional[int]


@dataclasses.dataclass(frozen=True)
class SplitSegment:
    """Represent one core time range plus extraction overlap for safe boundaries."""

    index: int
    core_start: float
    core_end: float
    audio_start: float
    audio_end: float

    @property
    def core_duration(self) -> float:
        """Return non-overlapping seconds counted toward video/run progress."""

        return max(0.0, self.core_end - self.core_start)

    @property
    def audio_duration(self) -> float:
        """Return extracted seconds, including context overlap at both edges."""

        return max(0.0, self.audio_end - self.audio_start)


@dataclasses.dataclass(frozen=True)
class SubtitleCue:
    """Store one finalized caption with absolute video timestamps."""

    start: float
    end: float
    text: str


@dataclasses.dataclass(frozen=True)
class AlignedToken:
    """Join a forced-aligned token with its punctuation-preserving text fragment."""

    start: float
    end: float
    fragment: str


@dataclasses.dataclass
class PerformanceTracker:
    """Track measured inference speed and provide increasingly accurate ETAs."""

    audio_seconds: float = 0.0
    inference_seconds: float = 0.0

    def record(self, audio_seconds: float, inference_seconds: float) -> None:
        """Add a completed inference sample to the run-wide speed estimate."""

        self.audio_seconds += max(0.0, audio_seconds)
        self.inference_seconds += max(0.0, inference_seconds)

    @property
    def real_time_factor(self) -> Optional[float]:
        """Return wall seconds per audio second, or None before calibration."""

        if self.audio_seconds <= 0.0:
            return None
        return self.inference_seconds / self.audio_seconds

    def estimate(self, remaining_audio_seconds: float) -> Optional[float]:
        """Estimate inference wall time for the supplied remaining audio."""

        factor = self.real_time_factor
        if factor is None:
            return None
        return max(0.0, remaining_audio_seconds) * factor


class Console:
    """Serialize timestamped pass, progress, warning, and error status output."""

    def __init__(self, quiet: bool = False) -> None:
        """Configure output verbosity, ANSI colors, and a thread-safe writer."""

        self.quiet = quiet
        self.use_color = sys.stderr.isatty() and "NO_COLOR" not in os.environ
        self._lock = threading.Lock()

    def _paint(self, text: str, code: str) -> str:
        """Apply ANSI color when attached to an interactive terminal."""

        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def _emit(self, level: str, message: str, color: str = "") -> None:
        """Write one complete status line without interleaving worker output."""

        if self.quiet and level not in {"ERROR", "WARN"}:
            return
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        label = f"[{level}]"
        if color:
            label = self._paint(label, color)
        with self._lock:
            print(f"[{timestamp}] {label} {message}", file=sys.stderr, flush=True)

    def pass_start(self, number: int, title: str, detail: str) -> None:
        """Announce a numbered processing pass and its concrete objective."""

        self._emit(f"PASS {number}/{PASS_COUNT}", f"{title} — {detail}", "1;36")

    def info(self, message: str) -> None:
        """Report a normal, durable progress event."""

        self._emit("INFO", message, "36")

    def success(self, message: str) -> None:
        """Report successful completion of a meaningful unit of work."""

        self._emit("DONE", message, "1;32")

    def warning(self, message: str) -> None:
        """Report a recoverable condition that may affect output or speed."""

        self._emit("WARN", message, "1;33")

    def error(self, message: str) -> None:
        """Report a failure to stderr while allowing the caller to choose policy."""

        self._emit("ERROR", message, "1;31")


class Heartbeat:
    """Emit elapsed time and a live ETA while a blocking model call is running."""

    def __init__(
        self,
        console: Console,
        label: str,
        interval: float,
        eta_provider: Optional[Callable[[float], Optional[float]]] = None,
    ) -> None:
        """Store heartbeat configuration without starting a background thread."""

        self.console = console
        self.label = label
        self.interval = max(2.0, interval)
        self.eta_provider = eta_provider
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0

    def __enter__(self) -> "Heartbeat":
        """Start a daemon reporter and return this context manager."""

        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Stop promptly and join the reporter when the blocking call finishes."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)

    def _run(self) -> None:
        """Wait between updates and report elapsed/estimated remaining time."""

        while not self._stop_event.wait(self.interval):
            elapsed = time.monotonic() - self._started_at
            eta = self.eta_provider(elapsed) if self.eta_provider else None
            eta_text = format_duration(eta) if eta is not None else "calibrating"
            self.console.info(
                f"{self.label} • elapsed {format_duration(elapsed)} • ETA {eta_text}"
            )


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the recursive input path and quality/performance tuning controls."""

    parser = argparse.ArgumentParser(
        prog="get_subs",
        description=(
            "Recursively generate adjacent SRT files with Qwen3-ASR-1.7B "
            "and Qwen3-ForcedAligner-0.6B."
        ),
    )
    parser.add_argument("directory", type=Path, help="Directory to scan recursively for videos.")
    parser.add_argument(
        "--language",
        default=None,
        help='Force a canonical language such as "English"; default is auto-detection.',
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional vocabulary/context hint for names, brands, or specialist terms.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Inference device. Auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=90.0,
        metavar="SECONDS",
        help="Target transcription chunk length (30-165; default: 90).",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="Audio context around chunk boundaries (0-5; default: 1.5).",
    )
    parser.add_argument(
        "--silence-db",
        type=float,
        default=-38.0,
        metavar="DB",
        help="Silence threshold used to find accurate split points (default: -38).",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.35,
        metavar="SECONDS",
        help="Minimum silence accepted as a split candidate (default: 0.35).",
    )
    parser.add_argument(
        "--max-cue-seconds",
        type=float,
        default=6.0,
        metavar="SECONDS",
        help="Maximum readable SRT cue duration (default: 6).",
    )
    parser.add_argument(
        "--max-cue-chars",
        type=int,
        default=84,
        metavar="COUNT",
        help="Maximum caption characters before a cue split (default: 84).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        metavar="COUNT",
        help="Maximum ASR decoder tokens per chunk (default: 512).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate completed SRT files; partial checkpoints always resume.",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
        help=f"Model cache directory (default: {DEFAULT_MODEL_CACHE}).",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Heartbeat interval during long model calls (default: 10).",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print warnings and errors.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    args = parser.parse_args(argv)

    if not 30.0 <= args.chunk_seconds <= 165.0:
        parser.error("--chunk-seconds must be between 30 and 165")
    if not 0.0 <= args.overlap_seconds <= 5.0:
        parser.error("--overlap-seconds must be between 0 and 5")
    if not 0.1 <= args.silence_duration <= 5.0:
        parser.error("--silence-duration must be between 0.1 and 5")
    if not 1.0 <= args.max_cue_seconds <= 15.0:
        parser.error("--max-cue-seconds must be between 1 and 15")
    if not 20 <= args.max_cue_chars <= 200:
        parser.error("--max-cue-chars must be between 20 and 200")
    if not 128 <= args.max_new_tokens <= 4096:
        parser.error("--max-new-tokens must be between 128 and 4096")
    if args.status_interval < 2.0:
        parser.error("--status-interval must be at least 2 seconds")

    return args


def format_duration(seconds: Optional[float]) -> str:
    """Format seconds as compact human-readable ETA/elapsed text."""

    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_bytes(byte_count: int) -> str:
    """Format a byte count using binary units for cache and disk reporting."""

    size = float(max(0, byte_count))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def command_path(name: str) -> Path:
    """Resolve a required external executable or raise a direct setup error."""

    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(
            f"Required command '{name}' was not found. On macOS run: brew install ffmpeg"
        )
    return Path(resolved)


def validate_directory(path: Path) -> Path:
    """Resolve and validate the user-supplied recursive scan root."""

    expanded = path.expanduser()
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Directory does not exist: {expanded}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"Path is not a directory: {resolved}")
    return resolved


def snapshot_size(path: Path) -> int:
    """Measure unique model files in a resolved Hugging Face snapshot."""

    total = 0
    seen: set[tuple[int, int]] = set()
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += stat.st_size
    return total


def ensure_model_snapshot(repo_id: str, cache_dir: Path, console: Console) -> Path:
    """Return a complete local model snapshot, downloading it when absent."""

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face runtime is missing; invoke get_subs through its launcher."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        resolved = Path(local_path).resolve()
        console.success(
            f"Model ready: {repo_id} ({format_bytes(snapshot_size(resolved))}, cached)"
        )
        return resolved
    except LocalEntryNotFoundError:
        pass

    console.info(
        f"Model is not installed: {repo_id}. Downloading now; the Hugging Face bars show file ETA."
    )
    started = time.monotonic()
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        local_files_only=False,
        max_workers=min(8, max(2, os.cpu_count() or 2)),
    )
    resolved = Path(local_path).resolve()
    console.success(
        f"Downloaded {repo_id} ({format_bytes(snapshot_size(resolved))}) in "
        f"{format_duration(time.monotonic() - started)}"
    )
    return resolved


def discover_videos(root: Path) -> list[Path]:
    """Find supported videos recursively, including every directory named subs."""

    discovered: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix.casefold() in VIDEO_EXTENSIONS:
            discovered.append(candidate)
    return sorted(discovered, key=lambda item: str(item).casefold())


def run_json_command(command: Sequence[str], description: str) -> dict[str, Any]:
    """Execute a metadata command and decode its JSON with useful diagnostics."""

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"{description} failed: {detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description} returned invalid JSON") from exc


def probe_video(path: Path, ffprobe: Path) -> VideoInfo:
    """Read duration and first-audio-stream metadata without decoding the video."""

    payload = run_json_command(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration:stream=index,codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        f"ffprobe for {path}",
    )
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("no audio stream")
    stream = streams[0]
    raw_duration = payload.get("format", {}).get("duration") or stream.get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("audio duration is unavailable") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise RuntimeError(f"invalid duration: {raw_duration}")

    sample_rate = stream.get("sample_rate")
    channels = stream.get("channels")
    return VideoInfo(
        path=path,
        output_path=path.with_suffix(".srt"),
        duration=duration,
        audio_codec=str(stream.get("codec_name") or "unknown"),
        sample_rate=int(sample_rate) if sample_rate else None,
        channels=int(channels) if channels else None,
    )


def validate_output_collisions(videos: Sequence[VideoInfo]) -> None:
    """Prevent two differently formatted videos from overwriting one stem.srt."""

    owners: dict[str, Path] = {}
    collisions: list[str] = []
    for video in videos:
        key = str(video.output_path.resolve()).casefold()
        previous = owners.get(key)
        if previous is not None and previous != video.path:
            collisions.append(f"{previous} and {video.path} -> {video.output_path}")
        owners[key] = video.path
    if collisions:
        joined = "\n  - ".join(collisions)
        raise RuntimeError(
            "Multiple videos would write the same SRT. Rename one of these files:\n  - " + joined
        )


def read_silence_intervals(
    video: VideoInfo,
    ffmpeg: Path,
    threshold_db: float,
    minimum_duration: float,
    console: Console,
) -> list[tuple[float, float]]:
    """Decode audio quickly and collect silence ranges for boundary-safe chunks."""

    if video.duration <= 1.0:
        return []
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        str(video.path),
        "-map",
        "0:a:0",
        "-af",
        f"silencedetect=noise={threshold_db:g}dB:d={minimum_duration:g}",
        "-f",
        "null",
        "-",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    intervals: list[tuple[float, float]] = []
    diagnostics: list[str] = []
    open_start: list[Optional[float]] = [None]
    silence_pattern = re.compile(
        r"silence_(start|end):\s*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[-+]?[0-9]+)?)",
        re.IGNORECASE,
    )

    def consume_diagnostics() -> None:
        """Parse FFmpeg silence messages while retaining a short failure tail."""

        assert process.stderr is not None
        for line in process.stderr:
            diagnostics.append(line.rstrip())
            if len(diagnostics) > 30:
                diagnostics.pop(0)
            match = silence_pattern.search(line)
            if match is None:
                continue
            marker, raw_value = match.groups()
            value = float(raw_value)
            if marker == "start":
                open_start[0] = value
            elif open_start[0] is not None and value > open_start[0]:
                intervals.append((open_start[0], value))
                open_start[0] = None

    diagnostic_thread = threading.Thread(target=consume_diagnostics, daemon=True)
    diagnostic_thread.start()
    last_reported_bucket = -1
    assert process.stdout is not None
    for line in process.stdout:
        key, separator, raw_value = line.strip().partition("=")
        if separator != "=" or key != "out_time_us":
            continue
        try:
            decoded_seconds = int(raw_value) / 1_000_000.0
        except ValueError:
            continue
        percent = min(100.0, decoded_seconds / video.duration * 100.0)
        bucket = int(percent // 10)
        if bucket > last_reported_bucket:
            last_reported_bucket = bucket
            console.info(f"Silence analysis {percent:5.1f}% • {video.path.name}")

    return_code = process.wait()
    diagnostic_thread.join(timeout=2.0)
    if open_start[0] is not None:
        intervals.append((open_start[0], video.duration))
    if return_code != 0:
        detail = "\n".join(diagnostics[-8:]).strip()
        raise RuntimeError(f"FFmpeg silence analysis failed: {detail}")
    return intervals


def build_segments(
    duration: float,
    silences: Sequence[tuple[float, float]],
    target_seconds: float,
    overlap_seconds: float,
) -> list[SplitSegment]:
    """Plan contiguous cores, preferring silence near target and enforcing 175s."""

    hard_maximum = 175.0
    minimum_useful = min(30.0, target_seconds * 0.45)
    silence_midpoints = sorted((start + end) / 2.0 for start, end in silences)
    boundaries = [0.0]
    cursor = 0.0

    split_threshold = min(target_seconds * 1.25, hard_maximum)
    while duration - cursor > split_threshold:
        remaining = duration - cursor
        ideal_length = target_seconds
        if remaining - ideal_length < minimum_useful:
            ideal_length = remaining / 2.0
        ideal = cursor + ideal_length
        latest = min(duration, cursor + hard_maximum)
        earliest = cursor + minimum_useful
        candidates = [
            point
            for point in silence_midpoints
            if earliest <= point <= latest and duration - point >= minimum_useful
        ]
        if candidates:
            cut = min(candidates, key=lambda point: abs(point - ideal))
        else:
            cut = min(ideal, latest)
        if cut <= cursor + 0.5:
            break
        boundaries.append(cut)
        cursor = cut

    boundaries.append(duration)
    segments: list[SplitSegment] = []
    for index, (core_start, core_end) in enumerate(zip(boundaries, boundaries[1:])):
        segments.append(
            SplitSegment(
                index=index,
                core_start=core_start,
                core_end=core_end,
                audio_start=max(0.0, core_start - overlap_seconds),
                audio_end=min(duration, core_end + overlap_seconds),
            )
        )
    return segments


def serialize_segment(segment: SplitSegment) -> dict[str, Any]:
    """Convert a split plan to stable JSON checkpoint primitives."""

    return dataclasses.asdict(segment)


def deserialize_segment(payload: dict[str, Any]) -> SplitSegment:
    """Rebuild a validated split segment from checkpoint JSON."""

    return SplitSegment(
        index=int(payload["index"]),
        core_start=float(payload["core_start"]),
        core_end=float(payload["core_end"]),
        audio_start=float(payload["audio_start"]),
        audio_end=float(payload["audio_end"]),
    )


def serialize_cue(cue: SubtitleCue) -> dict[str, Any]:
    """Convert one caption to checkpoint-safe JSON primitives."""

    return dataclasses.asdict(cue)


def deserialize_cue(payload: dict[str, Any]) -> SubtitleCue:
    """Rebuild one caption from a checkpoint while normalizing its types."""

    return SubtitleCue(
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload["text"]),
    )


def source_fingerprint(video: VideoInfo) -> dict[str, Any]:
    """Capture inexpensive source identity fields used to validate resume state."""

    stat = video.path.stat()
    return {
        "path": str(video.path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "duration_ms": int(round(video.duration * 1000.0)),
    }


def checkpoint_path(output_path: Path) -> Path:
    """Return the private sidecar used to resume an interrupted SRT write."""

    return output_path.with_name(f".{output_path.name}.get_subs.state.json")


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a text file atomically so readers never observe a torn SRT/state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Persist resume metadata before publishing the corresponding SRT snapshot."""

    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, serialized)


def load_checkpoint(path: Path) -> Optional[dict[str, Any]]:
    """Load a checkpoint or return None when no partial run exists."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read resume checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Resume checkpoint is not a JSON object: {path}")
    return payload


def checkpoint_matches(payload: dict[str, Any], video: VideoInfo) -> bool:
    """Confirm checkpoint schema and exact source identity before resuming."""

    return (
        payload.get("version") == CHECKPOINT_VERSION
        and payload.get("source") == source_fingerprint(video)
        and isinstance(payload.get("segments"), list)
        and isinstance(payload.get("cues"), list)
    )


def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert a non-negative floating timestamp to SRT's HH:MM:SS,mmm."""

    milliseconds = max(0, int(round(seconds * 1000.0)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt(cues: Sequence[SubtitleCue]) -> str:
    """Render ordered cues into standards-compatible UTF-8 SRT text."""

    blocks: list[str] = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n"
            f"{seconds_to_srt_timestamp(cue.start)} --> {seconds_to_srt_timestamp(cue.end)}\n"
            f"{cue.text}\n"
        )
    return "\n".join(blocks)


def write_srt(path: Path, cues: Sequence[SubtitleCue]) -> None:
    """Publish every completed chunk immediately as the requested stem.srt file."""

    atomic_write_text(path, render_srt(cues))


def extract_audio_segment(
    video: VideoInfo,
    segment: SplitSegment,
    destination: Path,
    ffmpeg: Path,
) -> None:
    """Decode one overlapped segment to model-native mono 16 kHz PCM WAV."""

    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-ss",
        f"{segment.audio_start:.3f}",
        "-i",
        str(video.path),
        "-t",
        f"{segment.audio_duration:.3f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not destination.exists():
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"Audio extraction failed: {detail}")


def resolve_device(requested: str, torch_module: Any) -> tuple[str, Any, str]:
    """Choose the fastest available accelerator and a memory-efficient dtype."""

    cuda_available = bool(torch_module.cuda.is_available())
    mps_available = bool(
        hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    )
    selected = requested
    if selected == "auto":
        selected = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if selected == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    if selected == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but PyTorch cannot access Apple Metal")

    if selected == "cuda":
        dtype = (
            torch_module.bfloat16
            if torch_module.cuda.is_bf16_supported()
            else torch_module.float16
        )
        return "cuda:0", dtype, "CUDA"
    if selected == "mps":
        return "mps", torch_module.float16, "Apple MPS"
    return "cpu", torch_module.float32, "CPU"


def clear_accelerator_cache(torch_module: Any, device_map: str) -> None:
    """Release unreferenced tensors after a video or recoverable model failure."""

    gc.collect()
    if device_map.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    elif device_map == "mps" and hasattr(torch_module, "mps"):
        torch_module.mps.empty_cache()


def load_qwen_pipeline(
    asr_path: Path,
    aligner_path: Path,
    requested_device: str,
    max_new_tokens: int,
    status_interval: float,
    console: Console,
) -> tuple[Any, Any, str]:
    """Load both required Qwen models once with host-optimized inference settings."""

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "The Qwen runtime is missing; invoke get_subs through its launcher."
        ) from exc

    device_map, dtype, device_label = resolve_device(requested_device, torch)
    # MPS SDPA currently falls back for parts of this model on some macOS /
    # PyTorch combinations. Eager attention keeps matrix multiplies on Metal;
    # CUDA still uses SDPA or FlashAttention when available.
    attention = "eager" if device_map == "mps" else "sdpa"
    if device_map.startswith("cuda") and importlib.util.find_spec("flash_attn") is not None:
        attention = "flash_attention_2"
    console.info(
        f"Loading both models on {device_label} with {str(dtype).removeprefix('torch.')} "
        f"and {attention} attention..."
    )
    model_kwargs = {
        "dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention,
    }
    with Heartbeat(console, "Loading Qwen models", status_interval):
        model = Qwen3ASRModel.from_pretrained(
            str(asr_path),
            forced_aligner=str(aligner_path),
            forced_aligner_kwargs=dict(model_kwargs),
            max_inference_batch_size=1,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )
    model.model.eval()
    if model.forced_aligner is None:
        raise RuntimeError("Qwen forced aligner did not initialize")
    model.forced_aligner.model.eval()
    console.success(f"Qwen3-ASR + ForcedAligner loaded on {device_label}")
    return model, torch, device_map


def kept_character(character: str) -> bool:
    """Mirror the aligner's punctuation-stripping definition for token matching."""

    return character == "'" or unicodedata.category(character).startswith(("L", "N"))


def normalized_character_stream(text: str) -> tuple[str, list[int]]:
    """Build a case-folded match stream with offsets back into original text."""

    normalized: list[str] = []
    offsets: list[int] = []
    for original_index, character in enumerate(text):
        if not kept_character(character):
            continue
        folded = unicodedata.normalize("NFKC", character).casefold()
        for normalized_character in folded:
            if kept_character(normalized_character):
                normalized.append(normalized_character)
                offsets.append(original_index)
    return "".join(normalized), offsets


def normalized_token(text: str) -> str:
    """Normalize one forced-alignment token for sequential transcript matching."""

    stream, _ = normalized_character_stream(text)
    return stream


def attach_transcript_fragments(
    transcript: str,
    alignment_items: Sequence[Any],
    audio_offset: float,
) -> list[AlignedToken]:
    """Restore ASR punctuation around aligned tokens while retaining timestamps."""

    stream, source_offsets = normalized_character_stream(transcript)
    match_cursor = 0
    spans: list[Optional[tuple[int, int]]] = []
    for item in alignment_items:
        needle = normalized_token(str(item.text))
        if not needle or not stream:
            spans.append(None)
            continue
        position = stream.find(needle, match_cursor)
        if position < 0:
            position = stream.find(needle, max(0, match_cursor - 12))
        if position < 0:
            spans.append(None)
            continue
        normalized_end = position + len(needle) - 1
        original_start = source_offsets[position]
        original_end = source_offsets[normalized_end] + 1
        spans.append((original_start, original_end))
        match_cursor = position + len(needle)

    tokens: list[AlignedToken] = []
    for index, item in enumerate(alignment_items):
        span = spans[index]
        if span is None:
            fragment = f"{item.text} "
        else:
            fragment_end = span[1]
            for later_span in spans[index + 1 :]:
                if later_span is not None:
                    fragment_end = later_span[0]
                    break
            else:
                fragment_end = len(transcript)
            fragment = transcript[span[0] : fragment_end]
        tokens.append(
            AlignedToken(
                start=audio_offset + max(0.0, float(item.start_time)),
                end=audio_offset + max(0.0, float(item.end_time)),
                fragment=fragment,
            )
        )
    return tokens


def clean_caption_text(text: str) -> str:
    """Collapse incidental whitespace while preserving language punctuation."""

    collapsed = " ".join(text.split()).strip()
    return re.sub(r"\s+([,.;:!?%。，；：！？])", r"\1", collapsed)


def wrap_caption(text: str, maximum_characters: int) -> str:
    """Wrap a caption to at most two balanced lines for common video players."""

    clean = clean_caption_text(text)
    line_width = max(10, int(math.ceil(maximum_characters / 2.0)))
    if len(clean) <= line_width:
        return clean
    if " " not in clean:
        midpoint = min(line_width, int(math.ceil(len(clean) / 2.0)))
        return clean[:midpoint] + "\n" + clean[midpoint:]
    lines = textwrap.wrap(
        clean,
        width=line_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) <= 2:
        return "\n".join(lines)
    first = lines[0]
    second = " ".join(lines[1:])
    return f"{first}\n{second}"


def tokens_to_cues(
    tokens: Sequence[AlignedToken],
    maximum_seconds: float,
    maximum_characters: int,
) -> list[SubtitleCue]:
    """Group word/character timestamps into readable punctuation-aware captions."""

    if not tokens:
        return []
    cues: list[SubtitleCue] = []
    group: list[AlignedToken] = []

    def flush_group() -> None:
        """Convert the current token group into one duration-safe subtitle cue."""

        nonlocal group
        if not group:
            return
        text = wrap_caption("".join(token.fragment for token in group), maximum_characters)
        if text:
            start = max(0.0, group[0].start)
            end = max(start + 0.2, group[-1].end)
            cues.append(SubtitleCue(start=start, end=end, text=text))
        group = []

    for index, token in enumerate(tokens):
        if group and token.start - group[-1].end >= 0.9:
            flush_group()
        group.append(token)
        current_text = clean_caption_text("".join(item.fragment for item in group))
        current_duration = max(0.0, group[-1].end - group[0].start)
        next_gap = 0.0
        if index + 1 < len(tokens):
            next_gap = tokens[index + 1].start - token.end
        punctuation_break = (
            bool(current_text)
            and current_text[-1] in STRONG_PUNCTUATION
            and current_duration >= 1.0
        )
        should_flush = (
            len(current_text) >= maximum_characters
            or current_duration >= maximum_seconds
            or len(group) >= 16
            or next_gap >= 0.9
            or punctuation_break
        )
        if should_flush:
            flush_group()
    flush_group()
    return cues


def filter_tokens_to_core(
    tokens: Sequence[AlignedToken],
    segment: SplitSegment,
    video_duration: float,
) -> list[AlignedToken]:
    """Assign overlapped words to exactly one contiguous core using midpoints."""

    selected: list[AlignedToken] = []
    final_segment = math.isclose(segment.core_end, video_duration, abs_tol=0.01)
    for token in tokens:
        midpoint = (token.start + token.end) / 2.0
        lower_match = midpoint >= segment.core_start - 0.001
        upper_match = (
            midpoint <= segment.core_end + 0.001
            if final_segment
            else midpoint < segment.core_end - 0.001
        )
        if lower_match and upper_match:
            selected.append(token)
    return selected


def merge_cues(existing: Sequence[SubtitleCue], additions: Sequence[SubtitleCue]) -> list[SubtitleCue]:
    """Append new cues while removing overlap duplicates and enforcing monotonic time."""

    merged = list(existing)
    for cue in additions:
        text_key = clean_caption_text(cue.text).casefold()
        if not text_key:
            continue
        if merged:
            previous = merged[-1]
            previous_key = clean_caption_text(previous.text).casefold()
            if previous_key == text_key and cue.start <= previous.end + 1.0:
                continue
            start = max(cue.start, previous.end + 0.001)
        else:
            start = cue.start
        end = max(start + 0.2, cue.end)
        merged.append(SubtitleCue(start=start, end=end, text=cue.text))
    return merged


def model_result_to_cues(
    result: Any,
    segment: SplitSegment,
    video_duration: float,
    maximum_seconds: float,
    maximum_characters: int,
) -> list[SubtitleCue]:
    """Convert Qwen ASR text plus forced alignment output into SRT-ready cues."""

    transcript = str(result.text or "").strip()
    if not transcript:
        return []
    alignment = result.time_stamps
    if alignment is None:
        raise RuntimeError("Qwen returned speech text without forced-alignment timestamps")
    items = list(getattr(alignment, "items", alignment))
    if not items:
        raise RuntimeError("Qwen forced aligner returned no timestamp items for non-empty speech")
    tokens = attach_transcript_fragments(transcript, items, segment.audio_start)
    selected = filter_tokens_to_core(tokens, segment, video_duration)
    return tokens_to_cues(selected, maximum_seconds, maximum_characters)


def make_checkpoint_payload(
    video: VideoInfo,
    segments: Sequence[SplitSegment],
    next_segment: int,
    cues: Sequence[SubtitleCue],
    languages: Sequence[str],
) -> dict[str, Any]:
    """Create the complete durable state required for interruption-safe resume."""

    return {
        "version": CHECKPOINT_VERSION,
        "source": source_fingerprint(video),
        "segments": [serialize_segment(segment) for segment in segments],
        "next_segment": next_segment,
        "cues": [serialize_cue(cue) for cue in cues],
        "languages": list(languages),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def transcribe_video(
    video: VideoInfo,
    model: Any,
    torch_module: Any,
    device_map: str,
    ffmpeg: Path,
    args: argparse.Namespace,
    console: Console,
    performance: PerformanceTracker,
    audio_after_video: float,
) -> tuple[str, int]:
    """Transcribe one video chunk-by-chunk and publish SRT after every chunk."""

    state_path = checkpoint_path(video.output_path)
    saved_state = load_checkpoint(state_path)
    if video.output_path.exists() and saved_state is None and not args.overwrite:
        console.info(f"Skipping completed output: {video.output_path}")
        return "skipped", 0

    if saved_state is not None and checkpoint_matches(saved_state, video):
        segments = [deserialize_segment(item) for item in saved_state["segments"]]
        next_segment = int(saved_state.get("next_segment", 0))
        cues = [deserialize_cue(item) for item in saved_state["cues"]]
        languages = [str(item) for item in saved_state.get("languages", [])]
        write_srt(video.output_path, cues)
        console.info(
            f"Resuming {video.path.name} at chunk {next_segment + 1}/{len(segments)} "
            f"with {len(cues)} cues already durable"
        )
    else:
        if saved_state is not None:
            console.warning(
                f"Discarding stale partial state because the source or schema changed: {state_path}"
            )
        console.info(
            f"Analyzing silence for accurate chunk boundaries: {video.path.name} "
            f"({format_duration(video.duration)})"
        )
        silences = read_silence_intervals(
            video,
            ffmpeg,
            args.silence_db,
            args.silence_duration,
            console,
        )
        segments = build_segments(
            video.duration,
            silences,
            args.chunk_seconds,
            args.overlap_seconds,
        )
        next_segment = 0
        cues: list[SubtitleCue] = []
        languages: list[str] = []
        initial_state = make_checkpoint_payload(video, segments, 0, cues, languages)
        save_checkpoint(state_path, initial_state)
        write_srt(video.output_path, cues)
        console.info(
            f"Planned {len(segments)} chunk(s) around {len(silences)} silence interval(s); "
            f"live output is {video.output_path}"
        )

    if next_segment >= len(segments):
        write_srt(video.output_path, cues)
        state_path.unlink(missing_ok=True)
        return "generated", len(cues)

    completed_core = sum(segment.core_duration for segment in segments[:next_segment])
    with tempfile.TemporaryDirectory(prefix="get_subs-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for segment in segments[next_segment:]:
            wav_path = temporary_root / f"chunk-{segment.index + 1:05d}.wav"
            console.info(
                f"Chunk {segment.index + 1}/{len(segments)} • extract "
                f"{format_duration(segment.audio_duration)} from {video.path.name}"
            )
            extract_audio_segment(video, segment, wav_path, ffmpeg)
            predicted = performance.estimate(segment.audio_duration)

            def chunk_eta(elapsed: float) -> Optional[float]:
                """Estimate remaining time inside the current blocking inference call."""

                if predicted is None:
                    return None
                return max(0.0, predicted - elapsed)

            console.info(
                f"Chunk {segment.index + 1}/{len(segments)} • Qwen3-ASR then "
                "Qwen3-ForcedAligner"
            )
            inference_started = time.monotonic()
            with Heartbeat(
                console,
                f"Chunk {segment.index + 1}/{len(segments)} inference",
                args.status_interval,
                chunk_eta,
            ):
                with torch_module.inference_mode():
                    result = model.transcribe(
                        audio=str(wav_path),
                        context=args.context,
                        language=args.language,
                        return_time_stamps=True,
                    )[0]
            inference_elapsed = time.monotonic() - inference_started
            performance.record(segment.audio_duration, inference_elapsed)
            if result.language and result.language not in languages:
                languages.append(str(result.language))
            additions = model_result_to_cues(
                result,
                segment,
                video.duration,
                args.max_cue_seconds,
                args.max_cue_chars,
            )
            cues = merge_cues(cues, additions)
            completed_core += segment.core_duration
            next_segment = segment.index + 1
            checkpoint = make_checkpoint_payload(
                video,
                segments,
                next_segment,
                cues,
                languages,
            )
            save_checkpoint(state_path, checkpoint)
            write_srt(video.output_path, cues)
            wav_path.unlink(missing_ok=True)

            video_percent = min(100.0, completed_core / video.duration * 100.0)
            speed = (
                segment.audio_duration / inference_elapsed if inference_elapsed > 0.0 else 0.0
            )
            remaining_audio = max(0.0, video.duration - completed_core) + max(
                0.0, audio_after_video
            )
            run_eta = performance.estimate(remaining_audio)
            console.success(
                f"Chunk {segment.index + 1}/{len(segments)} written • {len(additions)} new / "
                f"{len(cues)} total cues • video {video_percent:.1f}% • "
                f"inference {format_duration(inference_elapsed)} ({speed:.2f}x realtime) • "
                f"run ETA {format_duration(run_eta)}"
            )

    write_srt(video.output_path, cues)
    state_path.unlink(missing_ok=True)
    language_text = ", ".join(languages) if languages else "no speech detected"
    console.success(
        f"Finished {video.output_path} • {len(cues)} cues • language {language_text}"
    )
    clear_accelerator_cache(torch_module, device_map)
    return "generated", len(cues)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Coordinate provisioning, recursive discovery, inference, and final reporting."""

    args = parse_arguments(argv)
    console = Console(quiet=args.quiet)
    run_started = time.monotonic()
    try:
        console.pass_start(1, "Preflight", "validating input, FFmpeg, cache, and disk")
        root = validate_directory(args.directory)
        ffmpeg = command_path("ffmpeg")
        ffprobe = command_path("ffprobe")
        cache_dir = args.model_cache.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(cache_dir).free
        console.info(
            f"Input {root} • FFmpeg {ffmpeg} • cache {cache_dir} • "
            f"free disk {format_bytes(free_bytes)}"
        )
        if free_bytes < 12 * 1024**3:
            console.warning(
                "Less than 12 GiB is free; both Qwen models and temporary files may not fit."
            )

        console.pass_start(2, "Model availability", "checking/downloading both required models")
        asr_path = ensure_model_snapshot(ASR_REPOSITORY, cache_dir, console)
        aligner_path = ensure_model_snapshot(ALIGNER_REPOSITORY, cache_dir, console)

        console.pass_start(3, "Recursive discovery", "including nested directories named subs")
        paths = discover_videos(root)
        if not paths:
            console.success(f"No supported video files found under {root}")
            return 0
        console.info(f"Discovered {len(paths)} candidate video file(s)")

        console.pass_start(4, "Media probing", "reading audio streams and total duration")
        videos: list[VideoInfo] = []
        for index, path in enumerate(paths, start=1):
            try:
                video = probe_video(path, ffprobe)
            except RuntimeError as exc:
                console.warning(f"Skipping {path}: {exc}")
                continue
            videos.append(video)
            console.info(
                f"Probed {index}/{len(paths)} • {path.relative_to(root)} • "
                f"{format_duration(video.duration)} • {video.audio_codec}"
            )
        if not videos:
            console.success("No discovered video contains a readable audio stream")
            return 0
        validate_output_collisions(videos)
        total_audio = sum(video.duration for video in videos)
        console.success(
            f"Ready: {len(videos)} video(s), {format_duration(total_audio)} total audio"
        )

        console.pass_start(5, "Runtime loading", "placing ASR and aligner on the fastest device")
        model, torch_module, device_map = load_qwen_pipeline(
            asr_path,
            aligner_path,
            args.device,
            args.max_new_tokens,
            args.status_interval,
            console,
        )

        console.pass_start(6, "Transcription", "writing each stem.srt after every completed chunk")
        performance = PerformanceTracker()
        completed_audio = 0.0
        generated = 0
        skipped = 0
        failed: list[tuple[Path, str]] = []

        for index, video in enumerate(videos, start=1):
            console.info(
                f"Video {index}/{len(videos)} • {video.path.relative_to(root)} • "
                f"{format_duration(video.duration)}"
            )

            try:
                outcome, _ = transcribe_video(
                    video,
                    model,
                    torch_module,
                    device_map,
                    ffmpeg,
                    args,
                    console,
                    performance,
                    total_audio - completed_audio - video.duration,
                )
            except Exception as exc:
                failed.append((video.path, str(exc)))
                console.error(
                    f"Failed {video.path}; partial SRT/checkpoint retained for resume: {exc}"
                )
                clear_accelerator_cache(torch_module, device_map)
            else:
                if outcome == "skipped":
                    skipped += 1
                else:
                    generated += 1
            completed_audio += video.duration

        elapsed = time.monotonic() - run_started
        console.success(
            f"Run complete in {format_duration(elapsed)} • generated {generated} • "
            f"skipped {skipped} • failed {len(failed)}"
        )
        if failed:
            for path, reason in failed:
                console.error(f"{path}: {reason}")
            return 1
        return 0
    except KeyboardInterrupt:
        console.warning("Interrupted; completed chunks are durable and the next run will resume.")
        return 130
    except RuntimeError as exc:
        console.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
