"""Generate a YouTube Shorts title, description, and tags from a clip transcript via local Ollama.

Also runnable as a CLI so the web UI can offer title options at thumbnail-pick
time (before the slow render), without importing the pipeline:
    python metadata.py --titles 5 --source-title "Legend" --srt movie.srt --ranges "3:20-3:30"
"""

import argparse
import json
import os
import re
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

# Default credit line for the YouTube edit-style description. The channel wants
# these to read like a hand-made After Effects edit — the tool is NEVER named.
DEFAULT_EDITING_PROGRAM = "Adobe After Effects 2024"

_FAIR_USE_DISCLAIMER = (
    "© Copyright\n\n"
    "All the videos, songs, images, and graphics used in the video belong to their "
    "respective owners and I or this channel don't claim any rights over them.\n\n"
    "Copyright Disclaimer Under Section 107 of the Copyright Act 1976, allowance is made "
    "for fair use for purposes such as news reporting, teaching, scholarship, and research. "
    "Fair use is a use permitted by copyright statutes that might otherwise be infringing. "
    "Non-profit, educational, or personal use tips the balance in favor of fair use.\n\n"
    "Thanks for Watching...!!❤️!!"
)
# The only non-localhost call the pipeline makes on its own is this Wikipedia
# lookup (edge-tts in tts.py is the other). Set CLIPPER_NO_WEB_CONTEXT=1 to skip
# it entirely and stay fully offline apart from Ollama — the LLM then leans on
# its own knowledge of the film for the description.
_SKIP_WEB_CONTEXT = os.getenv("CLIPPER_NO_WEB_CONTEXT", "").strip() not in ("", "0", "false", "False")
DEFAULT_MODEL = "nemotron-3-super:cloud"
FALLBACK_MODEL = "gemma4:e2b"

_PROMPT_TEMPLATE = """You are writing YouTube Shorts metadata for a clip from "{source_title}".

Movie/show background (use this to enrich the description):
---
{movie_context}
---

Transcript of the clip (dialogue only):
---
{transcript}
---

Write metadata for this as a YouTube Short. The "title" will be burned into the
video itself as an on-screen text card, so it must be SHORT and punchy.

The description must be genuinely informative (aim for 4-6 sentences):
- Open with 2-3 sentences about the movie/show itself — genre, premise, the main
  characters, and tone — drawn from the background above
- Add 1-2 sentences about what happens in THIS specific moment and why it lands
- Naturally mention "{source_title}" for search

IMPORTANT: "{source_title}" is the movie/show name only. Do NOT put any
release/format jargon in the title or description — no resolution (1080p, 4K,
2160p), no IMAX, Blu-ray, HDR, Dolby Vision, codecs (x265, HEVC), or audio
channel tags (5.1). Those describe the video file, not the film.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "title": "max 40 characters, plain text only (no emoji, no quotes, no release/format words), all caps not needed, a punchy hook that creates curiosity",
  "description": "4-6 sentences as described above — NO hashtags here, those are added separately",
  "tags": ["10 to 15 short lowercase keyword tags as a list of strings, include the show/movie name and any character names mentioned, plus genre/mood tags for YouTube Shorts discovery"]
}}
"""


_TITLES_PROMPT_TEMPLATE = """You are writing YouTube Shorts titles for a clip from "{source_title}".

Movie/show background (use this for accurate character and plot references):
---
{movie_context}
---

Transcript of the clip (dialogue only):
---
{transcript}
---

Write {count} DIFFERENT title options for this Short. Each will be burned into the
video as an on-screen text card, so each must be SHORT and punchy.

Make the options genuinely varied in angle — not {count} rewordings of one idea.
Cover a spread of these: a curiosity hook, a direct quote-style line, a
character-focused line (name the character if the background or transcript makes
it clear), and a stakes/payoff line.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "titles": [
    {{"title": "max 40 characters, plain text only (no emoji, no quotes)", "angle": "3-5 words naming the angle, e.g. 'curiosity hook'"}}
  ]
}}
"""


def generate_titles(
    transcript: str,
    source_title: str = "",
    model: str = DEFAULT_MODEL,
    count: int = 5,
) -> list[dict]:
    """Propose `count` alternative on-screen titles for the clip.

    Same Wikipedia grounding as generate_metadata, so the options can name
    characters correctly instead of guessing from dialogue alone. Returns a list
    of {"title", "angle"} dicts; the caller picks one and passes it to main.py
    as --title.
    """
    title = source_title or "this movie/show"
    movie_context = fetch_movie_context(title) if source_title else ""
    if not movie_context:
        movie_context = f"No background found. Use general knowledge about '{title}'."

    prompt = _TITLES_PROMPT_TEMPLATE.format(
        transcript=transcript.strip(),
        source_title=title,
        movie_context=movie_context,
        count=count,
    )

    try:
        raw = _call_ollama(model, prompt)
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
        if model == FALLBACK_MODEL:
            raise
        print(f"      ! {model} unavailable ({type(exc).__name__}), falling back to {FALLBACK_MODEL}")
        raw = _call_ollama(FALLBACK_MODEL, prompt)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama did not return valid JSON: {raw!r}") from exc

    out: list[dict] = []
    for item in data.get("titles") or []:
        # Tolerate a bare list of strings — smaller models drop the object shape.
        if isinstance(item, str):
            text, angle = item, ""
        elif isinstance(item, dict):
            text, angle = str(item.get("title") or "").strip(), str(item.get("angle") or "").strip()
        else:
            continue
        if text:
            out.append({"title": text[:60], "angle": angle})
    return out[:count]


def fetch_movie_context(source_title: str) -> str:
    """Fetch a brief Wikipedia summary for the movie/show to ground the description.

    Best-effort and fail-fast: a short timeout means a blocked or slow route
    (e.g. traffic funnelled through a VPN/exit node) degrades gracefully to no
    context rather than stalling the pipeline. Disable with CLIPPER_NO_WEB_CONTEXT=1.
    """
    if _SKIP_WEB_CONTEXT:
        return ""
    try:
        encoded = requests.utils.quote(source_title)
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}",
            timeout=4,
            headers={"User-Agent": "movie-shorts-clipper/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "")
            return extract[:700] if extract else ""
    except Exception:
        pass
    return ""


def _call_ollama(model: str, prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "format": "json", "stream": False},
        timeout=180,
    )
    if response.status_code == 404:
        raise requests.HTTPError(
            f"Model '{model}' not found in Ollama. Run `ollama list` to see installed models.",
            response=response,
        )
    response.raise_for_status()
    return response.json()["response"]


def generate_metadata(transcript: str, source_title: str = "", model: str = DEFAULT_MODEL) -> dict:
    title = source_title or "this movie/show"
    movie_context = fetch_movie_context(title) if source_title else ""
    if not movie_context:
        movie_context = f"No background found. Use general knowledge about '{title}'."

    prompt = _PROMPT_TEMPLATE.format(
        transcript=transcript.strip(),
        source_title=title,
        movie_context=movie_context,
    )

    try:
        raw = _call_ollama(model, prompt)
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
        if model == FALLBACK_MODEL:
            raise
        print(f"      ! {model} unavailable ({type(exc).__name__}), falling back to {FALLBACK_MODEL}")
        raw = _call_ollama(FALLBACK_MODEL, prompt)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama did not return valid JSON: {raw!r}") from exc

    data.setdefault("title", "")
    data.setdefault("description", "")
    data.setdefault("tags", [])

    # Guarantee hashtags are appended to the description regardless of LLM compliance
    if data["tags"]:
        hashtags = " ".join(f"#{t.replace(' ', '').replace('-', '')}" for t in data["tags"][:10])
        desc = data["description"].rstrip()
        if "#" not in desc:
            data["description"] = desc + "\n\n" + hashtags
        else:
            # LLM added some; ensure our full tag list is still appended cleanly
            data["description"] = desc.rstrip() + "\n" + hashtags

    return data


def _to_hashtag(tag: str) -> str:
    """Turn a keyword/phrase into a CamelCase hashtag ('stranger things' -> '#StrangerThings')."""
    words = re.split(r"[\s_\-]+", str(tag).strip())
    camel = "".join(w[:1].upper() + w[1:] for w in words if w)
    camel = re.sub(r"[^A-Za-z0-9]", "", camel)
    return f"#{camel}" if camel else ""


def clean_music_name(name: str) -> str:
    """Best-effort clean of a music filename stem into a readable credit."""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", str(name))          # strip extension if present
    stem = re.sub(r"^\s*\d+[\.\)\-\s]+", "", stem)             # leading track number
    stem = stem.replace("_", " ").replace("-", " - ")
    return re.sub(r"\s+", " ", stem).strip()


def build_edit_description(
    meta: dict,
    movie_label: str = "",
    movie_tag: str = "",
    music: str = "",
    editing_program: str = DEFAULT_EDITING_PROGRAM,
    disclaimer: bool = True,
) -> str:
    """Format the LLM metadata into the YouTube "edit" description layout:

        <hook title>  #edit
        Movie: <movie>
        Music: <track>
        Editing Program: Adobe After Effects 2024

        #Hashtag #Block ...

        © Copyright ... (fair-use disclaimer)

    The tool is never named. `movie_label` carries the release year for the
    credit line; `movie_tag` (no year) seeds the hashtag block.
    """
    title = (meta.get("title") or "").strip()
    tags = list(meta.get("tags") or [])

    # Hashtags: movie name first, then the LLM tags, then evergreen edit tags.
    seed = ([movie_tag] if movie_tag else []) + tags + ["edit", "shorts", "fyp", "viral"]
    hashtags: list[str] = []
    seen: set[str] = set()
    for t in seed:
        h = _to_hashtag(t)
        if h and h.lower() not in seen:
            seen.add(h.lower())
            hashtags.append(h)

    lines: list[str] = []
    lines.append(f"{title}  #edit" if title else "#edit")
    lines.append("")
    if movie_label:
        lines.append(f"Movie: {movie_label}")
    if music:
        lines.append(f"Music: {music}")
    lines.append(f"Editing Program: {editing_program}")
    lines.append("")
    if hashtags:
        lines.append(" ".join(hashtags[:15]))
    if disclaimer:
        lines.append("")
        lines.append(_FAIR_USE_DISCLAIMER)
    return "\n".join(lines)


def _transcript_from_srt(srt: str, ranges_str: str) -> str:
    """Dialogue for the chosen ranges, straight from the subtitle file.

    Deferred imports: this is the CLI path only, so `import metadata` stays free
    of the pipeline's heavier modules.
    """
    from pathlib import Path

    from clip_selector import parse_ranges
    from subtitles import cues_for_ranges, cues_to_text, parse_srt

    return cues_to_text(cues_for_ranges(parse_srt(Path(srt)), parse_ranges(ranges_str)))


def _transcript_from_clip(clip: str, whisper_model: str, language: str, source_title: str) -> str:
    """Dialogue transcribed straight from the already-cut clip with Whisper.

    Used for title options when the user has NO subtitle file selected — the clip
    is only a few seconds, so this is quick, and it uses the same model the render
    will, so no extra model download happens (bandwidth cost = 0 once cached).
    Deferred import keeps faster-whisper out of the module's import cost.
    """
    from pathlib import Path

    from transcriber import transcribe, words_to_text

    words = transcribe(Path(clip), model_size=whisper_model, language=language, source_title=source_title)
    return words_to_text(words)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Shorts metadata or title options.")
    parser.add_argument("--titles", type=int, default=5, help="How many title options to propose (default: 5)")
    parser.add_argument("--source-title", default="", help="Movie/show name, used for web context")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--srt", default=None, help="Subtitle file to pull the clip's dialogue from")
    parser.add_argument("--ranges", default=None, help="Ranges to pull from --srt, e.g. '3:20-3:30'")
    parser.add_argument("--transcript", default=None, help="Use this transcript text directly instead of --srt")
    parser.add_argument("--clip", default=None,
                        help="Transcribe this already-cut clip with Whisper (used when there is no subtitle file)")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model for --clip (default: medium)")
    parser.add_argument("--language", default="en", help="Transcription language for --clip (default: en)")
    args = parser.parse_args()

    if args.transcript:
        transcript = args.transcript
    elif args.srt and args.ranges:
        transcript = _transcript_from_srt(args.srt, args.ranges)
    elif args.clip:
        transcript = _transcript_from_clip(args.clip, args.whisper_model, args.language, args.source_title)
    else:
        print("Error: need --transcript, --srt+--ranges, or --clip", file=sys.stderr)
        sys.exit(1)

    if not transcript.strip():
        print("Error: no dialogue found for those ranges", file=sys.stderr)
        sys.exit(1)

    titles = generate_titles(transcript, source_title=args.source_title,
                             model=args.model, count=args.titles)
    print(json.dumps({"titles": titles}))


if __name__ == "__main__":
    main()
