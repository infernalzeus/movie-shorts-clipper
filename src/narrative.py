"""Generate a long-form NARRATIVE script for a movie: an ordered list of story
segments, each a clip start-time plus the narrator's line for that clip.

Where recap.py picks a fast montage of standalone beats, this writes a CONNECTED
story — a movie-recap-channel voiceover that retells the whole plot in order.
The narration adds original storytelling; copyright clearance and monetization
eligibility remain separate review requirements.

The clip LENGTH is driven by the narration at render time (each clip is shown
for as long as its narration takes to read), so here we only pick the start
time and write the line.

Used two ways:
  - imported by main.py (narrative mode) via suggest_narrative()
  - shelled out to by the web UI:  python narrative.py --srt <path> --json
"""

import argparse
import json
import sys
from pathlib import Path

import requests

from clip_selector import parse_timestamp
from metadata import DEFAULT_MODEL, FALLBACK_MODEL, fetch_movie_context
from scenes import _call_ollama, _condense, _fmt_ts, _loads_lenient
from subtitles import parse_srt
from editorial import text_issues

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ~150 spoken words per minute; a passage is ~2-3 sentences (~35 words ≈ 14s).
_WORDS_PER_MIN = 150
_DEFAULT_TARGET = 540.0   # ~9 minutes of narration

# How many segments to ask for in ONE model call. A feature film needs 30-40
# segments, and asking a small local model for all of them at once does not
# work: it writes a dozen good ones, then degenerates into repetition or
# whitespace and never closes the JSON. Worse, everything it did write covers
# only the first fifth of the film. Generating in time-windowed parts keeps each
# response small enough to finish AND guarantees the ending gets covered.
_SEGMENTS_PER_CALL = 6
_MAX_PARTS = 12
# Most dialogue one call should carry, in condensed characters (~4 chars/token).
# 12k chars is ~3k tokens, which leaves ample room inside a modest context
# window on a 12 GB card instead of demanding a 32k allocation per request.
_MAX_WINDOW_CHARS = 12000

_PROMPT_TEMPLATE = """You are the narrator of a long-form YouTube movie-recap video for "{source_title}" — the kind that retells the entire plot as a gripping story, in order, over clips from the film.

Movie background:
---
{movie_context}
---

The movie's dialogue, condensed from the subtitles, in time order. Each line is:
[start - end] dialogue

---
{condensed}
---

{part_block}DURATION BUDGET: this part must contain at most {word_budget} spoken words TOTAL across all segments. Use fewer sentences when needed. Do not add padding to meet a segment count.

Write the NARRATION SCRIPT as an ordered list of {count} segments. Each segment is one moment in the story: a clip start time (from the timestamps above) and the narrator's line spoken over that clip.

This must read as ONE CONNECTED STORY, not disconnected facts:
- Tell the plot in chronological order from beginning to end — setup, characters, the inciting incident, rising action, the twists, the climax, and the ending.
- Each narration line is 1-3 concise sentences that MOVE THE STORY FORWARD and connect to the line before it (use "then", "but", "meanwhile", cause and effect). The viewer should never feel lost.
- Narrate what is happening and why it matters; you may paraphrase or set up a key line, but do NOT just transcribe the dialogue.
- Keep the narrator's voice clear and engaging — present tense, active, a little dramatic, but not cheesy.
- Cover the WHOLE span of dialogue you were given, end to end, spreading the segments evenly across it. Do not stall on the opening.
- SPEAKER ATTRIBUTION: subtitles never say who is speaking, and several characters speak in any
  given stretch — the lines are NOT all from the famous lead. Name a specific character ONLY when the
  dialogue itself makes it unmistakable (they are addressed by name, or the surrounding lines identify
  them). Otherwise use a role ("the captain", "the thief", "the reporter") or stay neutral. A WRONG
  NAME IS FAR WORSE THAN NO NAME: it is the single most visible error in a recap, and it makes the
  whole retelling untrustworthy.
- Describe ONLY events supported by the supplied dialogue. Movie background is context, not evidence of what happens in this window. Do not invent actions, motives, family relationships, or resolutions. When a speaker or action is uncertain, omit that detail rather than guess. This may be a small excerpt: do not force a whole-film climax or ending into it.

Respond with ONLY a JSON object in this exact shape:
{{
  "segments": [
    {{
      "start": "H:MM:SS",
      "narration": "the narrator's line spoken over this clip (1-3 concise sentences within the total word budget)"
    }}
  ]
}}
"""


def _windows(cues: list, n_parts: int) -> list[list]:
    """Split cues into n_parts contiguous windows of roughly equal DURATION.

    By time, not by cue count: dialogue-dense stretches would otherwise swallow
    several windows and the quiet third act would get none.
    """
    if n_parts <= 1:
        return [cues]
    span_start, span_end = cues[0].start, cues[-1].end
    step = max(1.0, (span_end - span_start) / n_parts)
    parts: list[list] = [[] for _ in range(n_parts)]
    for cue in cues:
        idx = min(n_parts - 1, max(0, int((cue.start - span_start) // step)))
        parts[idx].append(cue)
    return [p for p in parts if p]


def _part_block(i: int, n_parts: int, w_start: float, w_end: float, story_so_far: str) -> str:
    """The framing that turns one call into 'part i of n' of a connected script."""
    if n_parts <= 1:
        return ""
    lines = [
        f"This is PART {i} OF {n_parts} of the script, covering "
        f"{_fmt_ts(w_start)} to {_fmt_ts(w_end)} of the film. Only the dialogue for "
        f"this stretch is shown above; write only this part's segments.",
    ]
    if story_so_far:
        lines.append(f'The previous part ended with: "{story_so_far}"')
        lines.append("Continue straight on from there — do not re-introduce the film, "
                     "and do not recap what came before.")
    else:
        lines.append("This is the opening: set up the world and the characters.")
    if i == n_parts:
        lines.append("This is the FINAL part: carry the story through the climax to the ending.")
    return "\n".join(lines) + "\n\n"


def narration_budget(target_seconds):
    """Scale the script to duration, reserving time for inter-line gaps."""
    if not isinstance(target_seconds, (int, float)) or not 0 < target_seconds < float("inf"):
        raise ValueError("Target duration must be finite and positive")
    count = max(2, min(80, round(target_seconds / 14)))
    words = max(10, int(max(1, target_seconds - 0.6 - 0.35 * count - 0.8) * 135 / 60))
    return count, words


def suggest_narrative(
    srt_path: Path,
    source_title: str = "",
    model: str = DEFAULT_MODEL,
    target_seconds: float = _DEFAULT_TARGET,
    progress=None,
    region: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Return an ordered narration script: [{start, range, narration}, ...].

    `start` is seconds on the source timeline; `narration` is the narrator's line
    for that clip. Segments are chronological. The clip's end is decided at render
    time from how long the narration takes to read, so only `start` is fixed here.

    `region`: optional list of (start, end) windows. When given, only subtitle cues
    inside them are considered, so the script covers just that stretch of the movie
    (lets the user summarize a section in a separate render).

    The script is generated in time-windowed PARTS rather than one call — see
    _SEGMENTS_PER_CALL. Each part gets the previous part's closing line so the
    retelling still reads as one connected story across the seams.
    """
    cues = parse_srt(srt_path)
    if not cues:
        raise ValueError(f"No cues parsed from {srt_path}")

    if region:
        cues = [c for c in cues if any(c.start < r_end and c.end > r_start
                                       for r_start, r_end in region)]
        if not cues:
            raise ValueError("No subtitle cues fall within the requested span")

    title = source_title or srt_path.stem
    movie_context = fetch_movie_context(title) or (
        f"No background found. Use general knowledge about '{title}'."
    )

    # ~14s of narration per segment on average → count from the target length.
    count, total_word_budget = narration_budget(target_seconds)
    n_parts = max(1, min(_MAX_PARTS, -(-count // _SEGMENTS_PER_CALL)))

    # Splitting on segment count ALONE inverts the workload: a short target wants
    # few segments, so it made few windows, so each call had to swallow half the
    # film. On a 104-minute film a 120s target sent ~7,000 tokens per call while
    # a 600s target sent ~2,100. Split on the SOURCE size too, so no single call
    # carries more dialogue than the model can comfortably read.
    condensed_chars = len(_condense(cues))
    n_by_size = -(-condensed_chars // _MAX_WINDOW_CHARS)
    n_parts = max(1, min(_MAX_PARTS, max(n_parts, n_by_size)))

    parts = _windows(cues, n_parts)
    n_parts = len(parts)
    # Distribute the SAME number of segments over however many windows we ended
    # up with. A flat `max(2, ...)` floor would multiply the segment count by the
    # window count and blow the duration budget the caller asked for.
    base, extra = divmod(count, n_parts)
    per_part_counts = [max(1, base + (1 if i < extra else 0)) for i in range(n_parts)]

    movie_end = cues[-1].end
    segments: list[dict] = []
    prev_start = -1.0
    story_so_far = ""
    failures: list[str] = []

    for i, window in enumerate(parts, 1):
        w_start, w_end = window[0].start, window[-1].end
        prompt = _PROMPT_TEMPLATE.format(
            source_title=title,
            movie_context=movie_context,
            condensed=_condense(window),
            count=per_part_counts[i - 1],
            word_budget=max(10, total_word_budget // n_parts),
            part_block=_part_block(i, n_parts, w_start, w_end, story_so_far),
        )
        if progress:
            progress(f"part {i}/{n_parts} ({_fmt_ts(w_start)}-{_fmt_ts(w_end)})")

        try:
            raw = _call_ollama(model, prompt)
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            if model == FALLBACK_MODEL:
                raise
            print(f"! {model} unavailable ({type(exc).__name__}), falling back to {FALLBACK_MODEL}",
                  file=sys.stderr)
            raw = _call_ollama(FALLBACK_MODEL, prompt)

        # Lenient: a part that degenerated halfway still has usable segments in it.
        segs_raw = _loads_lenient(raw, "segments")
        if not segs_raw:
            failures.append(f"part {i}: nothing usable in {len(raw)} chars")
            continue

        kept = 0
        for item in segs_raw:
            try:
                start = parse_timestamp(str(item["start"]))
            except (KeyError, ValueError):
                continue
            narration = str(item.get("narration", "")).strip()
            problems = text_issues(narration)
            if problems:
                raise ValueError("Storytelling needs review before rendering: " + "; ".join(problems))
            if not narration or start < 0 or start >= movie_end:
                continue
            # Snap the start to the nearest cue start so the clip opens on a line.
            near = min(cues, key=lambda c: abs(c.start - start))
            if abs(near.start - start) <= 3.0:
                start = near.start
            if start <= prev_start:      # keep chronological, no repeats
                continue
            segments.append({
                "start": round(start, 1),
                "range": _fmt_ts(start),
                "narration": narration,
            })
            prev_start = start
            kept += 1
        if kept:
            story_so_far = segments[-1]["narration"]
        else:
            failures.append(f"part {i}: {len(segs_raw)} segments, none usable")

    if not segments:
        raise ValueError("the model produced no usable segments — "
                         + ("; ".join(failures) if failures else "empty response"))
    return segments


def estimate_duration(segments: list[dict]) -> float:
    """Rough spoken length of the whole script, in seconds (before TTS)."""
    words = sum(len(s.get("narration", "").split()) for s in segments)
    return words / _WORDS_PER_MIN * 60.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a long-form narrative script from an .srt file.")
    parser.add_argument("--srt", required=True, help="Path to the subtitle file")
    parser.add_argument("--title", default="", help="Movie title (default: derived from filename)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--target-seconds", type=float, default=_DEFAULT_TARGET,
                        help=f"Target narration length in seconds (default: {_DEFAULT_TARGET:.0f})")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.is_file():
        print(f"Error: SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)

    segments = suggest_narrative(srt_path, source_title=args.title, model=args.model,
                                 target_seconds=args.target_seconds)

    if args.json:
        print(json.dumps({"segments": segments}, ensure_ascii=False))
        return

    if not segments:
        print("No narrative segments produced.")
        return
    for i, s in enumerate(segments, 1):
        print(f"{i:2d}. [{s['range']}] {s['narration']}")
    print(f"\n{len(segments)} segments, ~{estimate_duration(segments) / 60:.1f} min of narration.")


if __name__ == "__main__":
    main()
