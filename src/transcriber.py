"""Transcribe a clip with faster-whisper and return word-level timestamps."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe(
    video_path: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = "en",
    source_title: str = "",
) -> list[Word]:
    """Run faster-whisper on video_path and return a flat list of word-level timestamps."""
    # Imported lazily so SRT-based runs never pay the faster-whisper/ctranslate2 load cost
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Seed Whisper with movie context so proper nouns and dialogue style bias its vocabulary.
    # Kept as a bare title (not a full sentence) — a sentence-form prompt like
    # "Transcript of dialogue from the movie ..." reads to Whisper as the opening line of
    # a transcript and gets regurgitated verbatim as fake dialogue.
    initial_prompt = f"Dialogue from '{source_title}'." if source_title else None
    # Normalised form of the prompt so we can detect and drop any segment that
    # just parrots it back (a common Whisper hallucination on quiet passages).
    prompt_norm = _norm(initial_prompt) if initial_prompt else ""

    segments, _info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        language=language,
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],  # retry with higher randomness on low-confidence segments
        initial_prompt=initial_prompt,
        # Without this, a single hallucinated prompt-echo becomes the context for
        # every later segment, so the seed sentence leaks onto EVERY caption.
        condition_on_previous_text=False,
        # Skip non-speech spans, where Whisper is most likely to invent the prompt
        # (or repeated filler) as phantom dialogue.
        vad_filter=True,
    )

    words = []
    for segment in segments:
        # Drop whole segments that are just the seed prompt echoed back.
        if prompt_norm and _norm(segment.text) == prompt_norm:
            continue
        for word in segment.words or []:
            words.append(Word(text=word.word.strip(), start=word.start, end=word.end))
    return words


def _norm(text: str) -> str:
    """Lowercase, strip punctuation/whitespace — for comparing text ignoring formatting."""
    import re
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def words_to_text(words: list[Word]) -> str:
    # Import here to avoid circular dependency (captions imports transcriber)
    from captions import _censor_word
    return " ".join(_censor_word(w.text) for w in words)
