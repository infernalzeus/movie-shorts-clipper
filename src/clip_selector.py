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


def cut_and_concat(video_path: Path, ranges: list[tuple[float, float]], output_path: Path,
                   fade_edges: bool = False) -> Path:
    """Cut each (start, end) range from video_path and concatenate them in order into output_path.

    fade_edges: fade each segment in from black and out to black at its own edges
    (~0.15s), so scenes dip smoothly between each other instead of hard-cutting.
    Segment durations are unchanged, so downstream caption timing stays in sync.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="movie-shorts-") as tmp:
        tmp_dir = Path(tmp)
        segment_paths = []

        for i, (start, end) in enumerate(ranges):
            segment_path = tmp_dir / f"segment_{i:02d}.mp4"
            fade_args: list[str] = []
            if fade_edges:
                seg = end - start
                fd = min(0.15, seg / 4)
                out_st = max(0.0, seg - fd)
                fade_args = [
                    "-vf", f"fade=t=in:st=0:d={fd:.3f},fade=t=out:st={out_st:.3f}:d={fd:.3f}",
                    "-af", f"afade=t=in:st=0:d={fd:.3f},afade=t=out:st={out_st:.3f}:d={fd:.3f}",
                ]
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(start),
                        "-i", str(video_path),
                        "-t", str(end - start),
                        *fade_args,
                        # -ac 2/-ar 48000: movies often carry 5.1 (or oddly
                        # tagged) audio; without normalizing here the unknown
                        # 6ch layout survives into burn.py's amix, where the
                        # AAC encoder refuses to open (exit -22).
                        # 256k on the first aac generation: this audio gets
                        # re-encoded twice more (burn mix, thumbnail append) —
                        # starting from the default 128k audibly smears by then.
                        # crf 16: this is an intermediate that gets re-encoded
                        # again at burn time — keep it near-transparent. veryfast
                        # keeps this near-lossless (crf sets quality; preset only
                        # trades speed vs size) while keeping the prepare phase
                        # snappy — medium made the cut slow enough to look hung.
                        "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                        "-c:a", "aac", "-b:a", "256k", "-ac", "2", "-ar", "48000",
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
        # Stream-COPY the concat, don't re-encode. Every segment above was cut
        # with identical encoder settings and starts on its own keyframe (each
        # `-ss` forces a fresh IDR), so the concat demuxer joins them cleanly with
        # `-c copy` — near-instant. Re-encoding here was pathologically slow for
        # recap mode (20-30+ segments → a multi-minute clip re-encoded at full
        # source resolution / libx264's slow default), and it's wasted work:
        # clip_raw.mp4 is an intermediate that burn.py re-encodes again anyway
        # (framing + captions + audio mix). Fall back to a fast re-encode only if
        # the copy concat ever fails (e.g. odd codec parameters).
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
        ]
        copy = subprocess.run(concat_cmd + ["-c", "copy", str(output_path)],
                              capture_output=True)
        if copy.returncode != 0:
            reencode = subprocess.run(
                concat_cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                              "-c:a", "aac", str(output_path)],
                capture_output=True,
            )
            if reencode.returncode != 0:
                stderr = reencode.stderr.decode(errors="replace") if reencode.stderr else ""
                raise RuntimeError(f"ffmpeg concat failed:\n{stderr}") from None

    return output_path
