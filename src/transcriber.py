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

    # Seed Whisper with movie context so proper nouns and dialogue style bias its vocabulary
    initial_prompt = f"Transcript of dialogue from the movie or show '{source_title}'." if source_title else None

    segments, _info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        language=language,
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],  # retry with higher randomness on low-confidence segments
        initial_prompt=initial_prompt,
    )

    words = []
    for segment in segments:
        for word in segment.words or []:
            words.append(Word(text=word.word.strip(), start=word.start, end=word.end))
    return words


def words_to_text(words: list[Word]) -> str:
    # Import here to avoid circular dependency (captions imports transcriber)
    from captions import _censor_word
    return " ".join(_censor_word(w.text) for w in words)
