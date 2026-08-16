"""Suggest clip-worthy scenes from a movie's .srt file via Ollama.

Condenses the subtitles into timestamped dialogue blocks, asks the LLM for the
most shorts-worthy moments, and snaps the suggested ranges to cue boundaries.

Used two ways:
  - imported by callers that want suggest_scenes()
  - shelled out to by the web UI:  python scenes.py --srt <path> --json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests

from clip_selector import parse_timestamp
from metadata import DEFAULT_MODEL, FALLBACK_MODEL, OLLAMA_URL, fetch_movie_context
from subtitles import Cue, parse_srt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_BLOCK_GAP = 4.0     # start a new dialogue block after this much silence
_BLOCK_MAX = 25.0    # ... or when a block grows past this many seconds
_PAD = 1.5           # padding added around the snapped cue boundaries

_PROMPT_TEMPLATE = """You are picking scenes from the movie "{source_title}" to publish as YouTube Shorts.

Movie background:
---
{movie_context}
---

Below is the movie's dialogue, condensed from the subtitles. Each line is:
[start - end] dialogue

---
{condensed}
---

Pick the {count} best self-contained moments for vertical short-form clips. Favor:
- dramatic confrontations, iconic or quotable exchanges, big reveals, sharp humor
- scenes that make sense without seeing the rest of the movie
- dialogue-dense stretches (the clip gets subtitles burned in)

Each scene must be 20 to 60 seconds long. Use the timestamps shown above.
Start and end on COMPLETE sentences — never begin or cut off in the middle of
a line of dialogue. Prefer a natural entry point (a question, a provocation,
a scene change) and a natural exit (a punchline, a verdict, a door slam).

Respond with ONLY a JSON object in this exact shape:
{{
  "scenes": [
    {{
      "start": "H:MM:SS",
      "end": "H:MM:SS",
      "title": "short punchy label for the scene (max 8 words)",
      "reason": "one sentence on why this works as a Short"
    }}
  ]
}}
"""


def _fmt_ts(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _condense(cues: list[Cue]) -> str:
    """Merge consecutive cues into timestamped dialogue blocks to keep the prompt compact."""
    lines: list[str] = []
    block: list[Cue] = []

    def flush() -> None:
        if block:
            text = " ".join(c.text for c in block)
            lines.append(f"[{_fmt_ts(block[0].start)} - {_fmt_ts(block[-1].end)}] {text}")

    for cue in cues:
        if block and (cue.start - block[-1].end > _BLOCK_GAP or cue.end - block[0].start > _BLOCK_MAX):
            flush()
            block = []
        block.append(cue)
    flush()
    return "\n".join(lines)


_SENTENCE_END_RE = re.compile(r"[.!?…][\"'”’)\]]*\s*$")
_MAX_SENTENCE_WALK = 16  # cues to extend in each direction hunting for sentence ends
_SENTENCE_GAP = 3.0      # a dialogue gap this long counts as a boundary too


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _snap_to_cues(cues: list[Cue], start: float, end: float) -> tuple[float, float]:
    """Expand [start, end] to complete-sentence boundaries, plus padding.

    Cue boundaries alone aren't enough — a sentence often spans several cues,
    so snapping to the overlapping cues still cuts mid-sentence. Walk outward:
    the start moves back until the PREVIOUS cue ends a sentence (so the clip
    opens on a sentence start), and the end moves forward until the last
    included cue ends a sentence. A long dialogue gap also counts as a
    boundary, and the walk is capped so unpunctuated subtitles can't drag the
    clip out indefinitely.
    """
    idx = [i for i, c in enumerate(cues) if c.end > start and c.start < end]
    if idx:
        i, j = idx[0], idx[-1]
        for _ in range(_MAX_SENTENCE_WALK):
            if i == 0 or _ends_sentence(cues[i - 1].text):
                break
            if cues[i].start - cues[i - 1].end >= _SENTENCE_GAP:
                break
            i -= 1
        for _ in range(_MAX_SENTENCE_WALK):
            if _ends_sentence(cues[j].text) or j + 1 >= len(cues):
                break
            if cues[j + 1].start - cues[j].end >= _SENTENCE_GAP:
                break
            j += 1
        start = min(start, cues[i].start)
        end = max(end, cues[j].end)
    movie_end = cues[-1].end if cues else end
    return max(0.0, start - _PAD), min(movie_end + 2.0, end + _PAD)


def _call_ollama(model: str, prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            # A whole movie's condensed dialogue is ~20K tokens — without this,
            # Ollama truncates at its small default context and the model
            # answers from a mangled prompt.
            "options": {"num_ctx": 32768},
        },
        timeout=600,
    )
    if response.status_code == 404:
        raise requests.HTTPError(
            f"Model '{model}' not found in Ollama. Run `ollama list` to see installed models.",
            response=response,
        )
    response.raise_for_status()
    return response.json()["response"]


def suggest_scenes(
    srt_path: Path,
    source_title: str = "",
    model: str = DEFAULT_MODEL,
    count: int = 5,
) -> list[dict]:
    """Return up to `count` suggested scenes: [{start, end, range, title, reason}, ...].

    start/end are seconds; range is a ready-to-use '--ranges' string chunk.
    """
    cues = parse_srt(srt_path)
    if not cues:
        raise ValueError(f"No cues parsed from {srt_path}")

    title = source_title or srt_path.stem
    movie_context = fetch_movie_context(title) or (
        f"No background found. Use general knowledge about '{title}'."
    )

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
        scenes_raw = json.loads(raw).get("scenes", [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama did not return valid JSON: {raw!r}") from exc

    movie_end = cues[-1].end
    scenes: list[dict] = []
    for item in scenes_raw:
        try:
            start = parse_timestamp(str(item["start"]))
            end = parse_timestamp(str(item["end"]))
        except (KeyError, ValueError):
            continue
        # LLMs (especially small local ones) hallucinate timestamps — validate hard.
        if end <= start or start >= movie_end:
            continue
        if end - start > 90:
            end = start + 60
        start, end = _snap_to_cues(cues, start, end)
        if end <= start or end - start < 8 or end - start > 100:
            continue
        scenes.append({
            "start": round(start, 1),
            "end": round(end, 1),
            "range": f"{_fmt_ts(start)}-{_fmt_ts(end)}",
            "title": str(item.get("title", "")).strip(),
            "reason": str(item.get("reason", "")).strip(),
        })
        if len(scenes) >= count:
            break
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest clip-worthy scenes from an .srt file.")
    parser.add_argument("--srt", required=True, help="Path to the subtitle file")
    parser.add_argument("--title", default="", help="Movie title (default: derived from filename)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--count", type=int, default=5, help="Number of scenes to suggest (default: 5)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.is_file():
        print(f"Error: SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)

    scenes = suggest_scenes(srt_path, source_title=args.title, model=args.model, count=args.count)

    if args.json:
        print(json.dumps({"scenes": scenes}, ensure_ascii=False))
        return

    if not scenes:
        print("No scenes suggested.")
        return
    for i, sc in enumerate(scenes, 1):
        print(f"{i}. [{sc['range']}] {sc['title']}")
        print(f"   {sc['reason']}")


if __name__ == "__main__":
    main()
