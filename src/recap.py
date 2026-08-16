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
_MIN_BEAT = 2.5    # allow short/shortest punchy lines too, as long as they're complete
_MAX_BEAT = 10.0
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

Pick {count} beats that together tell a CONNECTED story a viewer can follow, IN CHRONOLOGICAL ORDER. Cover the arc: the setup and main characters, the inciting incident, the major turning points, and — most importantly — build to the CLIMAX. It is fine if the most exciting material is early; pick the moments that actually carry the plot, wherever they fall.

PRIORITISE THE PAYOFF: the biggest reveal, twist, or final confrontation is the most important beat. The recap should build toward it and END on it — the LAST beat should be the decisive line of that moment (the ultimate thing said), not a quiet wind-down after it.

Rules for each beat:
- Each beat is a COMPLETE thought or exchange — a full statement, or a short back-and-forth (a line and its reaction) that MAKES SENSE ON ITS OWN. Never a sentence fragment.
- Aim for about 7 seconds each ({min_beat:.0f}-{max_beat:.0f}s) — long enough to land the exchange, not so long it drags. Start on the first word of a line and end on the last word of a line — never mid-sentence.
- The beats read back-to-back, so favor lines that CONNECT: each should follow from the last so the recap feels like one story, not disconnected clips.
- Chronological, non-overlapping. Favor plot-carrying lines (a reveal, a decision, a threat, a death, a twist) over filler or small talk.
- FEWER, meatier beats beat many tiny ones.

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
    avg_beat = (min_beat + max_beat) / 2   # ~7.5s
    count = max(4, min(20, round(target_seconds / avg_beat) + 2))

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
        end = min(movie_end + 1.0, end + 0.4)   # extra tail: subtitle end lags the audio
        # Do NOT trim the end back to max_beat — that chopped dialogue mid-sentence.
        # _snap_to_cues put the end on a complete-sentence boundary (with padding),
        # so the spoken line finishes; if that makes a beat run long, trim the
        # START forward to the nearest cue instead, keeping the end intact.
        if end - start > max_beat + 4:
            keep = [c for c in cues if c.start >= end - (max_beat + 2) and c.start < end]
            if keep:
                start = keep[0].start
        if end - start < min_beat or end - start > max_beat + 8:
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

    # Fallback: the LLM returned nothing usable (common with a small local model
    # or a narrow scoped region). Pick dialogue beats straight from the SRT
    # timestamps instead — spread evenly across the cues, each a short beat sized
    # by the cue's own duration. No LLM needed; dialogue sync = the SRT timing.
    if not beats:
        want = max(4, int(target_seconds / ((min_beat + max_beat) / 2)))
        step_n = max(1, len(cues) // want)
        for c in cues[::step_n]:
            s, e = _snap_to_cues(cues, c.start, min(c.end, c.start + max_beat))
            if e - s > max_beat:
                e = s + max_beat
            if e - s < min_beat or s < prev_end:
                continue
            beats.append({
                "start": round(s, 1),
                "end": round(e, 1),
                "range": f"{_fmt_ts(s)}-{_fmt_ts(e)}",
                "title": (c.text or "").strip()[:48],
                "reason": "auto (SRT timing)",
            })
            prev_end = e
            total += e - s
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
