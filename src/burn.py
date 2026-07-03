"""Burn an .ass subtitle file into a video with ffmpeg, optionally mixing in background music."""

import random
import subprocess
from pathlib import Path

_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a"}

# Available watermarks — add more entries here to extend the selectable list later.
WATERMARKS: dict[str, str] = {
    "mv-edits": "MV EDITS",
}


def _escape_filter_path(path: Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg -vf filtergraph argument."""
    p = path.resolve().as_posix()
    p = p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return p


def _watermark_filter(text: str, visible_until: float = 3.0) -> str:
    """Build a drawtext filter: bottom-centre, 50% translucent white, disappears at visible_until seconds."""
    safe_text = text.replace("'", "\\'").replace(":", "\\:")
    return (
        f"drawtext=text='{safe_text}'"
        f":fontsize=28"
        f":fontcolor=white"
        f":alpha=0.5"
        f":x=(w-tw)/2"
        f":y=h-th-40"
        f":enable='between(t,0,{visible_until})'"
    )


def pick_random_audio(audio_dir: Path) -> Path | None:
    """Return a random audio file from audio_dir, or None if the folder is empty."""
    if not audio_dir.is_dir():
        return None
    candidates = [f for f in audio_dir.iterdir() if f.suffix.lower() in _AUDIO_EXTS]
    return random.choice(candidates) if candidates else None


def burn_subtitles(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
    bg_music: Path | None = None,
    bg_volume: float = 0.10,
    watermark: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ass_arg = _escape_filter_path(ass_path)

    vf = f"ass='{ass_arg}'"
    if watermark:
        vf += f",{_watermark_filter(watermark)}"
    print(f"      [vf] {vf}")

    try:
        if bg_music:
            dur = _get_duration(video_path)
            filter_complex = (
                f"[1:a]aloop=loop=-1:size=2000000000,atrim=0:{dur},"
                f"volume={bg_volume}[bg];"
                f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(bg_music),
                "-filter_complex", filter_complex,
                "-vf", vf,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "libx264", "-c:a", "aac",
                str(output_path),
            ])
        else:
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", vf,
                "-c:v", "libx264", "-c:a", "copy",
                str(output_path),
            ])
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}):\n{stderr}") from None
    return output_path


def _run_ffmpeg(cmd: list) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _get_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())
