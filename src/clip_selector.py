"""Cut one or more timestamp ranges out of a source video and concatenate them into a single clip."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_FADE_DUR = 0.35  # seconds for fade-in / fade-out at clip boundaries


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
    """Cut each (start, end) range from video_path and concatenate them in order into output_path.

    When multiple ranges are given, each clip fades out to black and the next fades in,
    producing a smooth cross-cut transition.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(ranges)

    with tempfile.TemporaryDirectory(prefix="movie-shorts-") as tmp:
        tmp_dir = Path(tmp)
        segment_paths = []

        for i, (start, end) in enumerate(ranges):
            segment_path = tmp_dir / f"segment_{i:02d}.mp4"
            duration = end - start

            # Build fade filters when joining multiple clips
            vf_parts: list[str] = []
            af_parts: list[str] = []
            if n > 1 and duration > _FADE_DUR * 2 + 0.1:
                if i > 0:
                    vf_parts.append(f"fade=t=in:st=0:d={_FADE_DUR}")
                    af_parts.append(f"afade=t=in:st=0:d={_FADE_DUR}")
                if i < n - 1:
                    vf_parts.append(f"fade=t=out:st={duration - _FADE_DUR}:d={_FADE_DUR}")
                    af_parts.append(f"afade=t=out:st={duration - _FADE_DUR}:d={_FADE_DUR}")

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(video_path),
                "-t", str(duration),
            ]
            if vf_parts:
                cmd += ["-vf", ",".join(vf_parts)]
            if af_parts:
                cmd += ["-af", ",".join(af_parts)]
            cmd += [
                "-c:v", "libx264", "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                str(segment_path),
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            segment_paths.append(segment_path)

        if len(segment_paths) == 1:
            shutil.copy2(segment_paths[0], output_path)
            return output_path

        concat_list = tmp_dir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in segment_paths),
            encoding="utf-8",
        )
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

    return output_path
