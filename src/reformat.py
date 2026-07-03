"""Reformat a clip to a square by center-cropping then scaling to size×size."""

import subprocess
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
