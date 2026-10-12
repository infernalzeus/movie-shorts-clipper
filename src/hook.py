"""Pick a cold-open HOOK: the 1-3 most gripping short lines from a movie, shown
first as a teaser so a viewer wants to keep watching.

Reuses scenes.py's machinery (condensing, sentence snapping, the Ollama call,
lenient JSON). Kept separate from recap/narrative because a hook is a different
ask: not plot coverage, just the punchiest 1-5s moments.

Used two ways:
  - imported by main.py via suggest_hook()
  - shelled out to by the web UI:  python hook.py --srt <path> --json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import requests

from clip_selector import parse_timestamp
from metadata import DEFAULT_MODEL, FALLBACK_MODEL, fetch_movie_context
from scenes import _call_ollama, _condense, _fmt_ts_precise, _loads_lenient, _snap_to_cues
from subtitles import parse_srt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HOOK_MIN = 1.0
_HOOK_MAX = 5.0

_PROMPT_TEMPLATE = """You are cutting the COLD-OPEN HOOK for a short video about the movie "{source_title}" — the first few seconds that play before anything else, to grab the viewer.

Movie background:
---
{movie_context}
---

The movie's dialogue, condensed from the subtitles, in time order. Each line is:
[start - end] dialogue

---
{condensed}
---

Pick the {count} MOST GRIPPING short lines to play FIRST as a teaser — a shocking line, a threat, a twist, a big question, or a punchy declaration that makes someone want to watch. Together they should tease the gist of the story without spoiling the ending.
Return {count} DISTINCT clips, not just one example. When three are requested, aim for 10-15 seconds TOTAL across those three clips. Keep them about the same situation and use absolute source timestamps, never excerpt-relative times.

Rank by SCROLL-STOPPING POWER, not by chronology: the FIRST line you return must be the single most arresting one in the supplied dialogue window — the one that would stop a thumb mid-scroll in the first 2 seconds. Order the rest by impact after it. (Hook strength = shock + stakes + a question it makes the viewer need answered — the signals real recap channels open on.)

Rules:
- Each is a SHORT, COMPLETE line, {min_beat:.0f}-{max_beat:.0f} seconds. Never a fragment.
- Punchy and self-contained — it must hit even with no context.
- Prefer lines that raise a QUESTION or stakes over ones that answer them (an answer with no question doesn't hook).
- Use the timestamps shown above. They must stay inside the supplied dialogue window.

Respond with ONLY a JSON object in this exact shape:
{{
  "hook": [
    {{ "start": "H:MM:SS", "end": "H:MM:SS", "line": "the line" }}
  ]
}}
"""


def suggest_hook(
    srt_path: Path,
    source_title: str = "",
    model: str = DEFAULT_MODEL,
    count: int = 3,
    region: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Return up to `count` short hook clips: [{start, end, range, line}, ...]."""
    count = max(1, min(3, count))   # a cold-open is 1-3 lines, never more
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
    prompt = _PROMPT_TEMPLATE.format(
        source_title=title, movie_context=movie_context, condensed="\n".join(f"[{_fmt_ts_precise(c.start)} - {_fmt_ts_precise(c.end)}] {c.text}" for c in cues),
        count=count, min_beat=_HOOK_MIN, max_beat=_HOOK_MAX,
    )

    try:
        raw = _call_ollama(model, prompt)
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
        if model == FALLBACK_MODEL:
            raise
        print(f"! {model} unavailable ({type(exc).__name__}), falling back to {FALLBACK_MODEL}", file=sys.stderr)
        raw = _call_ollama(FALLBACK_MODEL, prompt)

    items = _loads_lenient(raw, "hook")
    # Local models sometimes return excerpt-relative timestamps (e.g. 0:00)
    # instead of the source timestamps. Never turn those into unrelated clips.
    def in_window(item):
        try:
            s = parse_timestamp(str(item["start"]))
            e = parse_timestamp(str(item["end"]))
            return (math.isfinite(s) and math.isfinite(e) and e > s
                    and s >= cues[0].start - 0.5 and e <= cues[-1].end + 0.5
                    and (not region or any(s >= a and e <= b for a, b in region)))
        except (KeyError, ValueError, TypeError):
            return False
    if sum(in_window(item) for item in items) < count:
        raw = _call_ollama(model, prompt + f"\nCORRECTION: Return {count} distinct clips. Use ABSOLUTE source timestamps exactly as shown above, NOT times relative to this excerpt. All clips must stay inside the supplied dialogue window.")
        items = _loads_lenient(raw, "hook")
    movie_end = cues[-1].end
    hooks: list[dict] = []
    seen: list[float] = []
    for item in items:
        if not in_window(item):
            continue
        try:
            start = parse_timestamp(str(item["start"]))
            end = parse_timestamp(str(item["end"]))
        except (KeyError, ValueError):
            continue
        if end <= start or start >= movie_end:
            continue
        start, end = _snap_to_cues(cues, start, end)
        if region:
            window = next(((a, b) for a, b in region if start < b and end > a), None)
            if window is None:
                continue
            start, end = max(start, window[0]), min(end, window[1])
        if end - start > _HOOK_MAX:          # a hook line must stay short
            end = start + _HOOK_MAX
        if end - start < _HOOK_MIN:
            continue
        if any(abs(start - s) < 1.0 for s in seen):   # no near-duplicates
            continue
        seen.append(start)
        hooks.append({
            "start": round(start, 1),
            "end": round(end, 1),
            "range": f"{_fmt_ts_precise(start)}-{_fmt_ts_precise(end)}",
            "line": str(item.get("line", "")).strip(),
        })
        if len(hooks) >= count:
            break
    return hooks


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick 1-3 cold-open hook clips from an .srt.")
    parser.add_argument("--srt", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.is_file():
        print(f"Error: SRT not found: {srt_path}", file=sys.stderr)
        sys.exit(1)
    hooks = suggest_hook(srt_path, source_title=args.title, model=args.model, count=args.count)
    if args.json:
        print(json.dumps({"hook": hooks, "ranges": "; ".join(h["range"] for h in hooks)}, ensure_ascii=False))
        return
    for i, h in enumerate(hooks, 1):
        print(f"{i}. [{h['range']}] {h['line']}")


if __name__ == "__main__":
    main()

