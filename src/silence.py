"""Detect near-silent gaps in a clip's audio and describe how to cut them out.

Backs the "remove blank spaces" option: sections where the audio is next to
zero (no dialogue, nothing meaningful) get cut from the clip before rendering,
and caption cue timings are remapped onto the shortened timeline via the
keep-intervals returned here.
"""

import re
import subprocess
from pathlib import Path

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(
    video_path: Path,
    noise_db: float = -35.0,
    min_silence: float = 1.0,
) -> list[tuple[float, float]]:
    """Run ffmpeg silencedetect over the clip's audio.

    Returns [(start, end)] silent intervals in seconds. Only gaps at least
    min_silence long count — short beats between lines are natural pacing,
    not blank space. A trailing silence that runs to EOF is returned with
    end=inf (the caller clamps to the clip duration).
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner",
            "-i", str(video_path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, errors="replace",
    )
    silences: list[tuple[float, float]] = []
    cur_start: float | None = None
    for line in proc.stderr.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            cur_start = max(0.0, float(m.group(1)))
            continue
        m = _SILENCE_END_RE.search(line)
        if m and cur_start is not None:
            silences.append((cur_start, float(m.group(1))))
            cur_start = None
    if cur_start is not None:
        silences.append((cur_start, float("inf")))
    return silences


def keep_intervals(
    duration: float,
    silences: list[tuple[float, float]],
    pad: float = 0.25,
    min_keep: float = 0.4,
) -> list[tuple[float, float]]:
    """Complement of the silent intervals: the [(start, end)] spans to keep.

    Each silence is shrunk by `pad` on both sides so speech onsets/tails are
    never clipped; a gap fully swallowed by its padding is kept as-is. Keeps
    shorter than min_keep are dropped — a sub-half-second sliver between two
    cuts reads as a glitch, not content.
    """
    keeps: list[tuple[float, float]] = []
    pos = 0.0
    for s, e in silences:
        end = duration if e == float("inf") else min(e, duration)
        cut_start = s + pad
        cut_end = end - pad
        if cut_end <= cut_start or cut_start >= duration:
            continue
        if cut_start > pos:
            keeps.append((pos, cut_start))
        pos = max(pos, min(cut_end, duration))
    if pos < duration:
        keeps.append((pos, duration))
    return [(s, e) for s, e in keeps if e - s >= min_keep]


def snap_ranges_to_speech(
    video_path: Path,
    ranges: list[tuple[float, float]],
    window: float = 2.5,
    pad: float = 0.12,
    noise_db: float = -33.0,
    min_gap: float = 0.14,
) -> list[tuple[float, float]]:
    """Snap each clip's start/end to a real speech GAP so cuts never land
    mid-word — the fix for lines chopped or bleeding in because the SUBTITLE
    timestamps don't match the actual spoken audio.

    Detects the audio's silences once, then for each range moves the start to the
    nearest speech ONSET (a silence end) and the end to the nearest speech OFFSET
    (a silence start) within `window` seconds. Verifying against the audio itself
    sidesteps subtitle-vs-audio drift (and Whisper's wording inaccuracy — we only
    use gap timing, never the words).
    """
    sils = detect_silences(video_path, noise_db=noise_db, min_silence=min_gap)
    if not sils:
        return ranges
    onsets = [e for _, e in sils if e != float("inf")]   # silence END  = speech starts
    offsets = [s for s, _ in sils]                        # silence START = speech ends
    out: list[tuple[float, float]] = []
    for s, e in ranges:
        on = [o for o in onsets if abs(o - s) <= window]
        off = [o for o in offsets if abs(o - e) <= window]
        ns = (min(on, key=lambda x: abs(x - s)) - pad) if on else s
        ne = (min(off, key=lambda x: abs(x - e)) + pad) if off else e
        ns = max(0.0, ns)
        out.append((round(ns, 2), round(ne, 2)) if ne - ns >= 0.5 else (s, e))
    return out


def remap_time(t: float, keeps: list[tuple[float, float]]) -> float:
    """Map a time on the original clip timeline onto the trimmed timeline.

    Times that fall inside a removed gap clamp to the end of the previous
    kept span (i.e. the moment the cut happens).
    """
    new = 0.0
    for s, e in keeps:
        if t < s:
            return new
        if t <= e:
            return new + (t - s)
        new += e - s
    return new
