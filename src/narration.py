"""Generate a scene narration script via Ollama — the transformative-content layer.

The narration is built as a sequence of time-anchored *beats*: the clip's
dialogue (with timestamps from the SRT) is grouped into beats, and the LLM writes
one analytical narration line per beat. Each line is anchored to when its beat
happens, so the voiceover spreads across the whole scene in sync with the action
instead of being read front-to-back and leaving dead air at the end.
"""

import json
import re
from dataclasses import dataclass

import requests

from metadata import FALLBACK_MODEL, OLLAMA_URL, fetch_movie_context
from subtitles import Cue

# Rough speaking rate of the edge-tts neural voices at default rate.
_WORDS_PER_SECOND = 2.4
# Narration should not start before the title card has cleared.
NARRATION_LEAD_IN = 2.2
# Fraction of each beat's time window the narration line should aim to fill —
# leaves a little breathing room before the next beat's line begins.
_FILL_FRACTION = 0.85

# Beat grouping: a new beat starts after this much silence between cues, or once
# a beat has run this long — so narration is re-anchored several times per clip.
_BEAT_GAP = 1.2
_BEAT_MAX = 9.0


@dataclass
class NarrationBeat:
    anchor: float       # when this line should start being spoken (clip seconds)
    window: float       # seconds available before the next beat's line
    max_words: int      # speak-time budget for this beat
    dialogue: str       # the dialogue spoken during this beat (LLM context)
    text: str = ""      # the narration line the LLM writes for this beat


_PROMPT_TEMPLATE = """You are the voice-over narrator for a YouTube Short cut from the film "{title}".
{context_block}{scene_block}
The clip is split into {n} beats, in time order. For each beat you are given the
dialogue spoken during it. Write ONE narration line per beat that ANALYSES the
moment — the tension, the power play, the subtext, what a character is really doing
beneath their words — drawing on what you know about "{title}" and its characters.

Rules:
- Insightful and vivid, not a flat description of the obvious. Present tense, third person.
- Stay consistent with the dialogue. Do NOT invent concrete physical facts the
  dialogue doesn't support (how many people are present, objects, the exact place).
- SPEAKER ATTRIBUTION: the subtitles never say who is speaking, and several
  characters speak in this scene — lines are NOT all from the famous lead. Name a
  specific character ONLY when the dialogue itself makes it unmistakable (they are
  addressed by name, or the surrounding dialogue identifies them). Otherwise refer
  to speakers by role ("the witness", "the prosecutor", "the judge") or neutrally.
  A wrong name is far worse than no name.
- Write ONE COMPLETE sentence per beat that fits within its word budget — finish
  the thought; never trail off. Fewer words that complete the idea beat a longer
  sentence that gets cut.
- No greetings, no channel talk, no hashtags, and don't quote the dialogue word-for-word.

Beats:
{beats_block}

Respond with ONLY a JSON object in this exact shape, with exactly {n} entries in order:
{{"beats": [{{"n": 1, "text": "..."}}, {{"n": 2, "text": "..."}}]}}
"""


def _call_ollama(model: str, prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            # Roomy enough for the surrounding-scene context blocks.
            "options": {"num_ctx": 16384},
        },
        timeout=300,
    )
    if response.status_code == 404:
        raise requests.HTTPError(
            f"Model '{model}' not found in Ollama. Run `ollama list` to see installed models.",
            response=response,
        )
    response.raise_for_status()
    return response.json()["response"]


def _fit_line(text: str, max_words: int) -> str:
    """Keep a narration line from wildly overrunning its beat WITHOUT chopping it
    mid-sentence.

    Small overshoots are left alone — the TTS step speeds the line up slightly to
    fit its window. Only when a line runs well past budget do we shorten it, and
    then only at a sentence boundary so it never ends mid-thought.
    """
    text = text.strip()
    words = text.split()
    # Allow a comfortable overshoot; tts.py absorbs the rest via a small speed-up.
    if len(words) <= max_words * 1.4:
        return text

    # Too long — keep whole sentences up to ~1.3x budget, dropping trailing ones.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    count = 0
    for sentence in sentences:
        n = len(sentence.split())
        if kept and count + n > max_words * 1.3:
            break
        kept.append(sentence)
        count += n
    result = " ".join(kept).strip()
    return result or " ".join(words[:max_words])


def build_beats(cues: list[Cue], clip_duration: float) -> list[NarrationBeat]:
    """Group timed cues into beats and size each beat's speak-time budget."""
    if not cues:
        return []

    clusters: list[list[Cue]] = []
    current: list[Cue] = []
    for cue in cues:
        if current and (
            cue.start - current[-1].end > _BEAT_GAP
            or cue.end - current[0].start > _BEAT_MAX
        ):
            clusters.append(current)
            current = []
        current.append(cue)
    if current:
        clusters.append(current)

    beats: list[NarrationBeat] = []
    for i, cluster in enumerate(clusters):
        anchor = cluster[0].start
        if i == 0:
            anchor = max(anchor, NARRATION_LEAD_IN)
        next_start = clusters[i + 1][0].start if i + 1 < len(clusters) else clip_duration
        window = max(1.5, next_start - anchor)
        max_words = max(4, int(window * _WORDS_PER_SECOND * _FILL_FRACTION))
        beats.append(NarrationBeat(
            anchor=anchor,
            window=window,
            max_words=max_words,
            dialogue=" ".join(c.text for c in cluster),
        ))
    return beats


def generate_narration_beats(
    cues: list[Cue],
    source_title: str,
    clip_duration: float,
    model: str,
    scene_before: str = "",
    scene_after: str = "",
) -> list[NarrationBeat]:
    """Return time-anchored narration beats with analytical lines written by the LLM.

    scene_before/scene_after: dialogue from the full subtitle file surrounding
    the clip — the setup usually reveals who is present and who is speaking,
    which the clip's own lines often don't."""
    beats = build_beats(cues, clip_duration)
    if not beats:
        return []

    context = fetch_movie_context(source_title)
    context_block = (
        f"\nBackground on the film (for names and context):\n{context}\n"
        if context else "\n"
    )
    scene_block = ""
    if scene_before:
        scene_block += (
            "\nDialogue from the movie right BEFORE this clip (scene setup — use it"
            f" to work out who is present and who is speaking):\n\"{scene_before}\"\n"
        )
    if scene_after:
        scene_block += f"\nDialogue right AFTER the clip:\n\"{scene_after}\"\n"

    beats_block = "\n".join(
        f'Beat {i + 1} (max {b.max_words} words) — dialogue: "{b.dialogue}"'
        for i, b in enumerate(beats)
    )
    prompt = _PROMPT_TEMPLATE.format(
        title=source_title,
        context_block=context_block,
        scene_block=scene_block,
        n=len(beats),
        beats_block=beats_block,
    )

    try:
        raw = _call_ollama(model, prompt)
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
        if model == FALLBACK_MODEL:
            raise
        print(f"      ! {model} unavailable ({type(exc).__name__}), falling back to {FALLBACK_MODEL}")
        raw = _call_ollama(FALLBACK_MODEL, prompt)

    try:
        parsed = json.loads(raw).get("beats", [])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama did not return valid JSON for narration: {raw!r}") from exc

    # Map the LLM's lines back onto our beats by 1-based index.
    by_index: dict[int, str] = {}
    for item in parsed:
        try:
            n = int(item["n"])
        except (KeyError, ValueError, TypeError):
            continue
        text = str(item.get("text", "")).strip()
        if text:
            by_index[n] = text

    out: list[NarrationBeat] = []
    for i, beat in enumerate(beats):
        text = by_index.get(i + 1, "")
        if not text:
            continue
        beat.text = _fit_line(text, beat.max_words)
        out.append(beat)

    if not out:
        raise ValueError("Ollama returned no usable narration lines")
    return out
