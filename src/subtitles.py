"""Parse a movie .srt file and map its cues onto a cut clip's timeline.

Replaces Whisper transcription when the source ships with a subtitle file:
cues overlapping the requested clip ranges are extracted, remapped to the
concatenated-clip timeline, and expanded into per-word timings so the
existing karaoke caption renderer works unchanged.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from transcriber import Word

_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")  # <i>…</i> HTML-ish tags and {\an8}-style ASS overrides
_SPEAKER_DASH_RE = re.compile(r"^\s*-\s*")


@dataclass
class Cue:
    start: float  # seconds
    end: float
    text: str


def _clean_cue_text(lines: list[str]) -> str:
    """Join cue lines into one string, dropping markup and SRT conventions."""
    cleaned = []
    for line in lines:
        line = _TAG_RE.sub("", line)
        line = _SPEAKER_DASH_RE.sub("", line)
        # Common SRT convention: quote marks wrapping voice-over narration lines
        line = re.sub(r"^\s*'", "", line)
        line = re.sub(r"([.!?,])'$", r"\1", line.strip())
        line = line.strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def parse_srt(srt_path: Path) -> list[Cue]:
    """Parse an .srt file into a list of Cues. Tolerant of BOM, CRLF, and stray blank lines."""
    raw = srt_path.read_text(encoding="utf-8-sig", errors="replace")
    cues: list[Cue] = []

    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        ts_idx = next((i for i, l in enumerate(lines) if _TIMESTAMP_RE.search(l)), None)
        if ts_idx is None:
            continue
        m = _TIMESTAMP_RE.search(lines[ts_idx])
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end   = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        text = _clean_cue_text(lines[ts_idx + 1:])
        if text and end > start:
            cues.append(Cue(start=start, end=end, text=text))

    cues.sort(key=lambda c: c.start)
    return cues


def find_srt_for_video(video_path: Path) -> Path | None:
    """Find a subtitle file for a movie: same-stem .srt first, then any .srt in
    the folder (preferring names that look English)."""
    exact = video_path.with_suffix(".srt")
    if exact.is_file():
        return exact
    candidates = sorted(video_path.parent.glob("*.srt"))
    if not candidates:
        return None
    for cand in candidates:
        if re.search(r"\b(english|eng|en)\b", cand.stem, re.IGNORECASE):
            return cand
    return candidates[0]


def cues_for_ranges(cues: list[Cue], ranges: list[tuple[float, float]]) -> list[Cue]:
    """Extract cues overlapping each (start, end) range and remap their times
    onto the concatenated-clip timeline (range 1 starts at 0, range 2 at
    len(range 1), ...). Cue times are clamped to the range edges."""
    remapped: list[Cue] = []
    offset = 0.0
    for range_start, range_end in ranges:
        for cue in cues:
            if cue.end <= range_start or cue.start >= range_end:
                continue
            start = max(cue.start, range_start) - range_start + offset
            end   = min(cue.end, range_end) - range_start + offset
            if end - start > 0.05:
                remapped.append(Cue(start=start, end=end, text=cue.text))
        offset += range_end - range_start
    return remapped


def cues_to_words(cues: list[Cue]) -> list[Word]:
    """Expand line-timed cues into per-word timings for the karaoke renderer.

    Each cue's duration is distributed across its words proportional to word
    length — a decent approximation of natural speech pacing.
    """
    words: list[Word] = []
    for cue in cues:
        tokens = cue.text.split()
        if not tokens:
            continue
        duration = cue.end - cue.start
        weights = [len(t) + 2 for t in tokens]  # +2 softens the bias toward long words
        total = sum(weights)
        t = cue.start
        for token, weight in zip(tokens, weights):
            dur = duration * weight / total
            words.append(Word(text=token, start=t, end=t + dur))
            t += dur
    return words


def cues_to_text(cues: list[Cue]) -> str:
    """Plain dialogue transcript of the cues (for LLM prompts)."""
    return " ".join(c.text for c in cues)


def cues_from_words(words: list[Word], gap: float = 0.6, max_dur: float = 6.0) -> list[Cue]:
    """Group Whisper word timings back into cue-like segments.

    Used for the narrated format when there is no SRT: the narration beats need
    timed dialogue chunks, which this reconstructs from the transcription.
    """
    cues: list[Cue] = []
    current: list[Word] = []
    for word in words:
        if current and (
            word.start - current[-1].end > gap
            or word.end - current[0].start > max_dur
        ):
            cues.append(Cue(
                start=current[0].start,
                end=current[-1].end,
                text=" ".join(w.text for w in current),
            ))
            current = []
        current.append(word)
    if current:
        cues.append(Cue(
            start=current[0].start,
            end=current[-1].end,
            text=" ".join(w.text for w in current),
        ))
    return cues
