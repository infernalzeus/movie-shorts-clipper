"""Generate a long-form NARRATIVE script for a movie: an ordered list of story
segments, each a clip start-time plus the narrator's line for that clip.

Where recap.py picks a fast montage of standalone beats, this writes a CONNECTED
story — a movie-recap-channel voiceover that retells the whole plot in order.
The narration is the transformative layer that makes the long-form video
monetizable; the clips are just what's shown while the narrator talks.

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
from scenes import _call_ollama, _condense, _fmt_ts
from subtitles import parse_srt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ~150 spoken words per minute; a passage is ~2-3 sentences (~35 words ≈ 14s).
_WORDS_PER_MIN = 150
_DEFAULT_TARGET = 540.0   # ~9 minutes of narration

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

Write the NARRATION SCRIPT as an ordered list of {count} segments. Each segment is one moment in the story: a clip start time (from the timestamps above) and the narrator's line spoken over that clip.

This must read as ONE CONNECTED STORY, not disconnected facts:
- Tell the plot in chronological order from beginning to end — setup, characters, the inciting incident, rising action, the twists, the climax, and the ending.
- Each narration line is 2-4 sentences that MOVE THE STORY FORWARD and connect to the line before it (use "then", "but", "meanwhile", cause and effect). The viewer should never feel lost.
- Narrate what is happening and why it matters; you may paraphrase or set up a key line, but do NOT just transcribe the dialogue.
- Keep the narrator's voice clear and engaging — present tense, active, a little dramatic, but not cheesy.
- Cover the WHOLE film end to end; don't stall on the opening.

Respond with ONLY a JSON object in this exact shape:
{{
  "segments": [
    {{
      "start": "H:MM:SS",
      "narration": "the narrator's line spoken over this clip (2-4 sentences)"
    }}
  ]
}}
"""


def suggest_narrative(
    srt_path: Path,
    source_title: str = "",
    model: str = DEFAULT_MODEL,
    target_seconds: float = _DEFAULT_TARGET,
) -> list[dict]:
    """Return an ordered narration script: [{start, range, narration}, ...].

    `start` is seconds on the source timeline; `narration` is the narrator's line
    for that clip. Segments are chronological. The clip's end is decided at render
    time from how long the narration takes to read, so only `start` is fixed here.
    """
    cues = parse_srt(srt_path)
    if not cues:
        raise ValueError(f"No cues parsed from {srt_path}")

    title = source_title or srt_path.stem
    movie_context = fetch_movie_context(title) or (
        f"No background found. Use general knowledge about '{title}'."
    )

    # ~14s of narration per segment on average → count from the target length.
    count = max(12, min(80, round(target_seconds / 14)))

    prompt = _PROMPT_TEMPLATE.format(
        source_title=title,
        movie_context=movie_context,
        condensed=_condense(cues),
        count=count,
    )

    try:
        raw = _call_ollama(model, prompt)
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
        if model == FALLBACK_MODEL:
            raise
        print(f"! {model} unavailable ({type(exc).__name__}), falling back to {FALLBACK_MODEL}", file=sys.stderr)
        raw = _call_ollama(FALLBACK_MODEL, prompt)

    try:
        segs_raw = json.loads(raw).get("segments", [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama did not return valid JSON: {raw!r}") from exc

    movie_end = cues[-1].end
    segments: list[dict] = []
    prev_start = -1.0
    for item in segs_raw:
        try:
            start = parse_timestamp(str(item["start"]))
        except (KeyError, ValueError):
            continue
        narration = str(item.get("narration", "")).strip()
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
