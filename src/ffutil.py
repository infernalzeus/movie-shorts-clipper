"""Shared ffprobe helpers."""

import subprocess
from pathlib import Path


def get_resolution(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Take only the first non-empty line — some files (e.g. IMAX dual-stream)
    # produce multiple lines, which makes a bare split("x") yield 3+ tokens.
    first = next((l.strip() for l in result.stdout.splitlines() if l.strip()), "")
    parts = first.split("x")
    if len(parts) < 2:
        raise ValueError(f"Unexpected ffprobe resolution output: {result.stdout!r}")
    return int(parts[0]), int(parts[1])
