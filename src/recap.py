"""Pick a chronological chain of short beats that summarize a whole movie's plot.

Where `scenes.py` finds the single best *standalone* scene, this finds the
*story spine* — a sequence of short moments (setup → inciting incident →
turning points → climax → resolution) in chronological order that, stitched
back to back, recap the film. It feeds the "Recap Montage" format.

Reuses scenes.py's machinery (condensing, sentence-boundary snapping, the
Ollama call, timestamp validation) so only the prompt and the selection rules
differ.

Used two ways:
  - imported by callers that want suggest_recap()
  - shelled out to by the web UI:  python recap.py --srt <path> --json
"""

import argparse
import json
import sys
from pathlib import Path

import requests

from clip_selector import parse_timestamp
from metadata import DEFAULT_MODEL, FALLBACK_MODEL, fetch_movie_context
from scenes import _call_ollama, _condense, _fmt_ts, _snap_to_cues
from subtitles import parse_srt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Recap beats are short — a beat is a single moment, not a whole scene.
_MIN_BEAT = 3.0
_MAX_BEAT = 12.0
_DEFAULT_TARGET = 120.0   # total body seconds to aim for (before the outro)

_PROMPT_TEMPLATE = """You are cutting a fast RECAP of the movie "{source_title}" — a montage that retells the whole plot in short beats, in order, the way movie-recap channels do.

Movie background:
---
{movie_context}
---

Below is the movie's dialogue, condensed from the subtitles, in time order. Each line is:
[start - end] dialogue

---
{condensed}
---

Pick {count} short beats that together tell the WHOLE story from start to finish, IN CHRONOLOGICAL ORDER. Cover the arc: the setup and main characters, the inciting incident, the major turning points and reversals, the climax, and the resolution. Spread the beats across the entire runtime — do NOT bunch them all near the start.

Rules for each beat:
- Each beat is a single punchy moment {min_beat:.0f}-{max_beat:.0f} seconds long. Prefer the shorter end — this is a fast montage.
- Beats must be in chronological order and must NOT overlap each other.
- Start and end on COMPLETE lines of dialogue — never cut off mid-sentence.
- Favor lines that carry the plot forward (a reveal, a decision, a threat, a death, a twist) over filler.

Respond with ONLY a JSON object in this exact shape:
{{
  "beats": [
    {{
      "start": "H:MM:SS",
      "end": "H:MM:SS",
      "title": "what happens in this beat (max 8 words)",
      "reason": "one clause on its role in the plot"
    }}
  ]
}}
"""


def suggest_recap(
    srt_path: Path,
    source_title: str = "",
    model: str = DEFAULT_MODEL,
    target_seconds: float = _DEFAULT_TARGET,
    min_beat: float = _MIN_BEAT,
    max_beat: float = _MAX_BEAT,
    region: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Return an ordered list of recap beats: [{start, end, range, title, reason}, ...].

    start/end are seconds on the SOURCE timeline; the beats are chronological,
    non-overlapping, and sum to roughly `target_seconds`. `range` is a
    ready-to-use '--ranges' chunk.

    `region`: optional list of (start, end) time windows. When given, only
    subtitle cues inside those windows are considered, so the recap is scoped to
    that part of the movie ("recap this stretch") instead of the whole film.
    """
    cues = parse_srt(srt_path)
    if not cues:
        raise ValueError(f"No cues parsed from {srt_path}")

    if region:
        cues = [c for c in cues if any(c.start < r_end and c.end > r_start
                                       for r_start, r_end in region)]
        if not cues:
            raise ValueError("No subtitle cues fall within the requested region(s)")

    title = source_title or srt_path.stem
    movie_context = fetch_movie_context(title) or (
        f"No background found. Use general knowledge about '{title}'."
    )

    # Aim a little high on the beat count — validation drops some, and we trim
    # to the duration budget at the end anyway.
    avg_beat = (min_beat + max_beat) / 2
    count = max(8, min(30, round(target_seconds / avg_beat) + 4))

    prompt = _PROMPT_TEMPLATE.format(
        source_title=title,
        movie_context=movie_context,
        condensed=_condense(cues),
        count=count,
        min_beat=min_beat,
        max_beat=max_beat,
    )

    try:
        raw = _call_ollama(model, prompt)
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
        if model == FALLBACK_MODEL:
            raise
        print(f"! {model} unavailable ({type(exc).__name__}), falling back to {FALLBACK_MODEL}", file=sys.stderr)
        raw = _call_ollama(FALLBACK_MODEL, prompt)

    try:
        beats_raw = json.loads(raw).get("beats", [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama did not return valid JSON: {raw!r}") from exc

    movie_end = cues[-1].end
    beats: list[dict] = []
    total = 0.0
    prev_end = 0.0  # enforce chronological, non-overlapping order

    for item in beats_raw:
        try:
            start = parse_timestamp(str(item["start"]))
            end = parse_timestamp(str(item["end"]))
        except (KeyError, ValueError):
            continue
        # Small local models hallucinate timestamps — validate hard.
        if end <= start or start >= movie_end:
            continue
        if end - start > max_beat * 1.6:
            end = start + max_beat
        start, end = _snap_to_cues(cues, start, end)
        # Snapping can widen a beat past the montage budget — trim the tail back.
        if end - start > max_beat:
            end = start + max_beat
        if end - start < min_beat or end - start > max_beat + 2:
            continue
        # Drop anything that runs backwards or overlaps the previous beat.
        if start < prev_end:
            continue
        beats.append({
            "start": round(start, 1),
            "end": round(end, 1),
            "range": f"{_fmt_ts(start)}-{_fmt_ts(end)}",
            "title": str(item.get("title", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
        })
        prev_end = end
        total += end - start
        if total >= target_seconds:
            break

    return beats


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick a chronological recap montage from an .srt file.")
    parser.add_argument("--srt", required=True, help="Path to the subtitle file")
    parser.add_argument("--title", default="", help="Movie title (default: derived from filename)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--target-seconds", type=float, default=_DEFAULT_TARGET,
                        help=f"Total body length to aim for (default: {_DEFAULT_TARGET:.0f})")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.is_file():
        print(f"Error: SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)

    beats = suggest_recap(srt_path, source_title=args.title, model=args.model,
                          target_seconds=args.target_seconds)

    if args.json:
        # A single '--ranges' string joining every beat, plus the structured list.
        ranges = "; ".join(b["range"] for b in beats)
        print(json.dumps({"beats": beats, "ranges": ranges}, ensure_ascii=False))
        return

    if not beats:
        print("No recap beats suggested.")
        return
    total = sum(b["end"] - b["start"] for b in beats)
    for i, b in enumerate(beats, 1):
        print(f"{i:2d}. [{b['range']}] {b['title']}")
        print(f"     {b['reason']}")
    print(f"\n{len(beats)} beats, ~{total:.0f}s total.")
    print("Ranges: " + "; ".join(b["range"] for b in beats))


if __name__ == "__main__":
    main()
