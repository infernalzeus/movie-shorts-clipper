"""Transcribe a clip with faster-whisper and return word-level timestamps."""

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


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
) -> list[Word]:
    """Run faster-whisper on video_path and return a flat list of word-level timestamps."""
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(str(video_path), word_timestamps=True)

    words = []
    for segment in segments:
        for word in segment.words or []:
            words.append(Word(text=word.word.strip(), start=word.start, end=word.end))
    return words


def words_to_text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)
