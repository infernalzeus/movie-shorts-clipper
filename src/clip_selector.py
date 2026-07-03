"""Cut one or more timestamp ranges out of a source video and concatenate them into a single clip."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def parse_timestamp(ts: str) -> float:
    """Parse 'mm:ss', 'hh:mm:ss', or 'm:ss.s' into seconds."""
    ts = ts.strip()
    match = _TIME_RE.match(ts)
    if not match:
        raise ValueError(f"Invalid timestamp: {ts!r} (expected mm:ss or hh:mm:ss)")
    hours, minutes, seconds = match.groups()
    hours = int(hours) if hours else 0
    return hours * 3600 + int(minutes) * 60 + float(seconds)


def parse_ranges(ranges_str: str) -> list[tuple[float, float]]:
    """Parse 'mm:ss-mm:ss; mm:ss-mm:ss' (semicolon or comma separated) into a list of (start, end) seconds."""
    ranges = []
    for chunk in re.split(r"[;,]", ranges_str):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"Invalid range: {chunk!r} (expected start-end)")
        start_str, end_str = chunk.split("-", 1)
        start, end = parse_timestamp(start_str), parse_timestamp(end_str)
        if end <= start:
            raise ValueError(f"Range end must be after start: {chunk!r}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("No valid ranges parsed")
    return ranges


def cut_and_concat(video_path: Path, ranges: list[tuple[float, float]], output_path: Path) -> Path:
    """Cut each (start, end) range from video_path and concatenate them in order into output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="movie-shorts-") as tmp:
        tmp_dir = Path(tmp)
        segment_paths = []

        for i, (start, end) in enumerate(ranges):
            segment_path = tmp_dir / f"segment_{i:02d}.mp4"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(start),
                        "-i", str(video_path),
                        "-t", str(end - start),
                        "-c:v", "libx264", "-c:a", "aac",
                        "-avoid_negative_ts", "make_zero",
                        str(segment_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="replace") if e.stderr else ""
                raise RuntimeError(f"ffmpeg segment {i} failed:\n{stderr}") from None
            segment_paths.append(segment_path)

        if len(segment_paths) == 1:
            shutil.copy2(segment_paths[0], output_path)
            return output_path

        concat_list = tmp_dir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in segment_paths),
            encoding="utf-8",
        )
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_list),
                    "-c:v", "libx264", "-c:a", "aac",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(f"ffmpeg concat failed:\n{stderr}") from None

    return output_path
