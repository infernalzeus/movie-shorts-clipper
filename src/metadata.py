"""Generate a YouTube Shorts title, description, and tags from a clip transcript via local Ollama."""

import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
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

The description must:
- Open with 1-2 sentences about the movie/show itself (genre, premise, tone) drawn from the background above
- Add 1 sentence about what makes this specific moment compelling
- Naturally mention "{source_title}" for search

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "title": "max 40 characters, plain text only (no emoji, no quotes), all caps not needed, a punchy hook that creates curiosity",
  "description": "3-4 sentences as described above — NO hashtags here, those are added separately",
  "tags": ["10 to 15 short lowercase keyword tags as a list of strings, include the show/movie name and any character names mentioned, plus genre/mood tags for YouTube Shorts discovery"]
}}
"""


def fetch_movie_context(source_title: str) -> str:
    """Fetch a brief Wikipedia summary for the movie/show to ground the description."""
    try:
        encoded = requests.utils.quote(source_title)
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}",
            timeout=10,
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
