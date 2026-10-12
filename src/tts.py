"""Synthesize the narration voiceover with edge-tts (free Microsoft neural voices).

Each narration beat is synthesized separately and placed at its own anchor time
(from narration.NarrationBeat) so the voiceover spreads across the whole clip in
sync with the action, rather than being read start-to-finish up front. A beat
whose speech runs longer than its time window is sped up (capped atempo) so it
doesn't collide with the next beat.
"""

import asyncio
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VOICE = "en-US-JennyNeural"  # female US narrator

# A few good narrator options surfaced in the web UI. Any edge-tts voice id works.
VOICES = [
    "en-US-ChristopherNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-AU-WilliamNeural",
]

_MIN_GAP = 0.15        # minimum silence between two narration lines (seconds)
_MAX_TEMPO = 1.35      # never speed speech up more than this to force a fit


@dataclass
class NarrationSegment:
    text: str
    start: float  # seconds on the clip timeline
    end: float


# Microsoft's endpoint rejects requests intermittently (HTTP 403 is the common
# one) and the same text succeeds unchanged moments later. A long-form script is
# one request per segment — 40+ of them — so without a retry the chance that at
# least one fails approaches certainty, and a single transient rejection throws
# away the whole job. Retry each line a few times with a growing pause.
_TTS_ATTEMPTS = 4
_TTS_BACKOFF = 2.0      # seconds, doubled each attempt


async def _synth_one(text: str, voice: str, out_path: Path) -> None:
    import edge_tts

    last: Exception | None = None
    for attempt in range(1, _TTS_ATTEMPTS + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(out_path))
            # A rejected request can still leave a zero-byte file behind, which
            # would survive to ffprobe as a corrupt input rather than an error.
            if out_path.is_file() and out_path.stat().st_size > 0:
                return
            raise RuntimeError("speech service returned an empty audio file")
        except Exception as exc:          # noqa: BLE001 - edge-tts raises bare Exception
            last = exc
            out_path.unlink(missing_ok=True)
            if attempt == _TTS_ATTEMPTS:
                break
            pause = _TTS_BACKOFF * (2 ** (attempt - 1))
            print(f"      ! speech attempt {attempt}/{_TTS_ATTEMPTS} failed "
                  f"({type(exc).__name__}); retrying in {pause:.0f}s", flush=True)
            await asyncio.sleep(pause)
    raise RuntimeError(
        f"speech generation failed after {_TTS_ATTEMPTS} attempts: {type(last).__name__}: {last}"
    ) from last


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def synthesize_script(
    lines: list[str],
    output_wav: Path,
    voice: str = DEFAULT_VOICE,
    gap: float = 0.35,
    lead_in: float = 0.6,
) -> list[NarrationSegment]:
    """Narration-LED synthesis: speak every line in order and report how long each
    one took, with nothing squeezed or dropped.

    This is the inverse of synthesize_narration(). There, the clip length is fixed
    and the speech is compressed (or discarded) to fit inside it. Here the speech
    is the fixed quantity and the caller derives the clip lengths from the returned
    segments — which is what a storytelling recap needs: the footage is shown for
    exactly as long as the narrator talks over it.

    Returns one NarrationSegment per line, back to back on a fresh timeline.
    """
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        raise RuntimeError("edge-tts is not installed. Run: pip install edge-tts") from None

    lines = [t.strip() for t in lines if t and t.strip()]
    if not lines:
        raise ValueError("synthesize_script got no lines")

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="narration-script-") as tmp:
        tmp_dir = Path(tmp)

        async def _synth_all() -> None:
            for i, text in enumerate(lines):
                await _synth_one(text, voice, tmp_dir / f"seg_{i:03d}.mp3")

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_synth_all())

        seg_paths = [tmp_dir / f"seg_{i:03d}.mp3" for i in range(len(lines))]
        durations = [_probe_duration(p) for p in seg_paths]

        segments: list[NarrationSegment] = []
        cursor = lead_in
        for text, dur in zip(lines, durations):
            segments.append(NarrationSegment(text=text, start=cursor, end=cursor + dur))
            cursor += dur + gap

        total = cursor
        inputs: list[str] = []
        filters: list[str] = []
        labels: list[str] = []
        for i, (seg, path) in enumerate(zip(segments, seg_paths)):
            inputs += ["-i", str(path)]
            delay_ms = int(seg.start * 1000)
            filters.append(f"[{i}:a]aresample=44100,adelay={delay_ms}|{delay_ms}[a{i}]")
            labels.append(f"[a{i}]")

        mix = (
            f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0,"
            f"apad,atrim=0:{total:.3f}[aout]"
        )
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", *inputs,
                    "-filter_complex", ";".join(filters + [mix]),
                    "-map", "[aout]", "-c:a", "pcm_s16le", str(output_wav),
                ],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(f"ffmpeg script assembly failed:\n{stderr}") from None

    return segments


def synthesize_narration(
    beats: list,
    output_wav: Path,
    clip_duration: float,
    voice: str = DEFAULT_VOICE,
    strict: bool = False,
) -> list[NarrationSegment]:
    """Build narration.wav (exactly clip_duration long) from time-anchored beats.

    `beats` is a list of narration.NarrationBeat (anchor, window, text). Each beat
    is spoken starting at its anchor time, sped up if needed to fit before the
    next beat, so the voiceover tracks the scene instead of front-loading.
    """
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "edge-tts is not installed. Run: pip install edge-tts"
        ) from None

    beats = [b for b in beats if getattr(b, "text", "").strip()]
    if not beats:
        raise ValueError("synthesize_narration got no beats with text")

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="narration-tts-") as tmp:
        tmp_dir = Path(tmp)

        async def _synth_all() -> None:
            for i, beat in enumerate(beats):
                await _synth_one(beat.text, voice, tmp_dir / f"seg_{i:02d}.mp3")

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_synth_all())

        seg_paths = [tmp_dir / f"seg_{i:02d}.mp3" for i in range(len(beats))]
        durations = [_probe_duration(p) for p in seg_paths]

        # Place each beat at its anchor, sped up to fit its window; nudge forward
        # only if a previous line would still be talking (avoids two voices at once).
        segments: list[NarrationSegment] = []
        tempos: list[float] = []
        keep_paths: list[Path] = []
        prev_end = 0.0
        for beat, path, dur in zip(beats, seg_paths, durations):
            start = beat.anchor if strict else max(beat.anchor, prev_end + _MIN_GAP)
            budget = max(1.0, beat.window - _MIN_GAP)
            tempo = 1.0
            if dur > budget:
                tempo = min(dur / budget, _MAX_TEMPO)
            eff_dur = dur / tempo
            if strict:
                validate_approved_timing(beat, start, eff_dur, prev_end, clip_duration)
            if start + eff_dur > clip_duration - 0.1:
                # Ran out of room near the end — drop rather than talk past the clip.
                print(f"      ! narration overflow — dropping: {beat.text[:60]!r}")
                continue
            segments.append(NarrationSegment(text=beat.text, start=start, end=start + eff_dur))
            tempos.append(tempo)
            keep_paths.append(path)
            prev_end = start + eff_dur

        if not segments:
            raise ValueError("no narration segments fit inside the clip")

        # Assemble: delay each segment to its start time, mix, pad/trim to clip length.
        inputs: list[str] = []
        filters: list[str] = []
        labels: list[str] = []
        for i, (seg, path, tempo) in enumerate(zip(segments, keep_paths, tempos)):
            inputs += ["-i", str(path)]
            delay_ms = int(seg.start * 1000)
            chain = f"[{i}:a]"
            if tempo > 1.02:
                chain += f"atempo={tempo:.4f},"
            filters.append(f"{chain}aresample=44100,adelay={delay_ms}|{delay_ms}[a{i}]")
            labels.append(f"[a{i}]")

        mix = (
            f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0,"
            f"apad,atrim=0:{clip_duration:.3f}[aout]"
        )
        filter_complex = ";".join(filters + [mix])

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    *inputs,
                    "-filter_complex", filter_complex,
                    "-map", "[aout]",
                    "-c:a", "pcm_s16le",
                    str(output_wav),
                ],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(f"ffmpeg narration assembly failed:\n{stderr}") from None

    return segments


def validate_approved_timing(beat, start, duration, previous_end, clip_duration):
    """Approved lines must stay at their anchors and inside their windows."""
    end = start + duration
    if (start < previous_end or end > start + beat.window - _MIN_GAP + 1e-6
            or end > clip_duration - 0.1):
        raise ValueError(f"Approved commentary does not fit at {beat.anchor:g}s: {beat.text!r}. Shorten the line or adjust its window.")
