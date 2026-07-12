"""Reformat a clip: square center-crop (classic) or the same square centred on
a black 9:16 canvas (narrated layout — title above, narration text below).
Also assembles the final upload video (prepending the thumbnail frame)."""

import os
import subprocess
import tempfile
from pathlib import Path


def crop_to_square(
    input_path: Path,
    output_path: Path,
    size: int = 1080,
) -> Path:
    """Center-crop the source video to a square and scale to size×size (default 1080×1080).

    Works for any input aspect ratio:
      - 1920×1080 (landscape) → crops the sides, keeps the middle 1080×1080
      - 3840×2160 (4K)        → crops to 2160×2160, scales down to 1080×1080
      - 1080×1920 (portrait)  → crops top/bottom, keeps the middle 1080×1080

    FFmpeg's crop filter centers by default when x/y are omitted.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_str = (
        f"crop=min(iw\\,ih):min(iw\\,ih),"
        f"scale={size}:{size}"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", filter_str,
            "-map", "0:v:0",
            "-map", "0:a:0?",       # optional — skipped if the clip has no audio
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def compose_vertical(
    input_path: Path,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
) -> tuple[Path, dict]:
    """Compose a 9:16 canvas: the classic square center-crop, vertically centred
    on a plain black background — title card above, narration text below.

    Returns (output_path, layout) where layout gives the pixel geometry the
    caption builder needs: {"width", "height", "video_top", "video_bottom"}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_top = (height - width) // 2  # square centred: e.g. 420..1500 on 1080x1920

    filter_str = (
        f"crop=min(iw\\,ih):min(iw\\,ih),"
        f"scale={width}:{width},"
        f"pad={width}:{height}:0:{video_top}:black"
    )

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-vf", filter_str,
                "-map", "0:v:0",
                "-map", "0:a:0?",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"ffmpeg vertical compose failed:\n{stderr}") from None

    layout = {
        "width": width,
        "height": height,
        "video_top": video_top,
        "video_bottom": video_top + width,
    }
    return output_path, layout


def append_still_image(
    video_path: Path,
    image_path: Path,
    output_path: Path,
    duration: float = 0.1,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """Append a `duration`-second freeze of image_path to the END of video_path.

    Bakes the thumbnail into the last fraction of the upload so it can be selected
    as the Short's thumbnail on YouTube mobile (which can't take a directly-
    uploaded thumbnail). Safe to call with output_path == video_path — it renders
    to a temp file first, then replaces the original.
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(output_path.parent))
    os.close(fd)
    tmp_out = Path(tmp_name)

    # Both inputs are 9:16; normalise fps/sar/format/audio so concat can join them.
    # Order is video-then-still so the thumbnail lands at the end.
    filter_complex = (
        f"[0:v]scale={width}:{height},setsar=1,fps=30,format=yuv420p[bv];"
        f"[1:v]scale={width}:{height},setsar=1,fps=30,format=yuv420p[tv];"
        f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[ba];"
        f"[2:a]aformat=sample_rates=44100:channel_layouts=stereo[sa];"
        f"[bv][ba][tv][sa]concat=n=2:v=1:a=1[v][a]"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-loop", "1", "-t", f"{duration}", "-i", str(image_path),
                "-f", "lavfi", "-t", f"{duration}", "-i", "anullsrc=r=44100:cl=stereo",
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "aac",
                str(tmp_out),
            ],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        tmp_out.unlink(missing_ok=True)
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"ffmpeg append-thumbnail failed:\n{stderr}") from None

    os.replace(tmp_out, output_path)
    return output_path
