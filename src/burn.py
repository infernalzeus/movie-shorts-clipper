"""Burn an .ass subtitle file into a video with ffmpeg, optionally mixing in background music."""

import random
import subprocess
from pathlib import Path

_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a"}

# Recap mix: the movie DIALOGUE is the lead and the music is a bed that ducks
# UNDER it (classic voice-over sidechain). Movie dialogue is quiet + dynamic
# (~-26 dB) while music is loudness-maximised (~-6 dB), so the dialogue is first
# lifted/evened with dynaudnorm and NEVER ducked; the music is keyed off the
# dialogue so it drops out of the way whenever someone is speaking.
# Gentle lift (makeup 5, ratio 3) — enough to bring quiet dialogue up without
# slamming peaks into the limiter (makeup 8 did that = the "tearing" distortion).
_RECAP_DIALOGUE_NORM = "acompressor=threshold=-26dB:ratio=3:attack=10:release=250:makeup=5"

# Available watermarks — add more entries here to extend the selectable list later.
WATERMARKS: dict[str, str] = {
    "mv-edits": "MV EDITS",
}


def _escape_filter_path(path: Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg -vf filtergraph argument."""
    p = path.resolve().as_posix()
    p = p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return p


def _watermark_filter(text: str, visible_until: float = 3.0) -> str:
    """Build a drawtext filter: bottom-centre, 50% translucent white, disappears at visible_until seconds."""
    safe_text = text.replace("'", "\\'").replace(":", "\\:")
    return (
        f"drawtext=text='{safe_text}'"
        f":fontsize=28"
        f":fontcolor=white"
        f":alpha=0.5"
        f":x=(w-tw)/2"
        f":y=h-th-40"
        f":enable='between(t,0,{visible_until})'"
    )


def pick_random_audio(audio_dir: Path) -> Path | None:
    """Return a random audio file from audio_dir, or None if the folder is empty."""
    if not audio_dir.is_dir():
        return None
    candidates = [f for f in audio_dir.iterdir() if f.suffix.lower() in _AUDIO_EXTS]
    return random.choice(candidates) if candidates else None


def burn_subtitles(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
    bg_music: Path | None = None,
    bg_volume: float = 0.10,
    bg_skip: float = 0.0,
    watermark: str | None = None,
    narration_audio: Path | None = None,
    narration_volume: float = 0.04,
    pre_filter: str | None = None,
    mirror: bool = False,
    music_duck: bool = False,
) -> Path:
    """Burn the .ass captions into the video and mix the audio layers:
    movie audio (full, unchanged) + narration voiceover at narration_volume,
    laid on top + optional looped bg music.

    pre_filter: framing filters (square crop / vertical canvas) applied before
    the captions, so framing and burning happen in ONE encode instead of two —
    one less lossy x264 generation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ass_arg = _escape_filter_path(ass_path)

    # hflip goes FIRST so only the footage is mirrored — the ass captions and the
    # watermark are added afterwards, so text stays the right way round.
    vf_parts = ["hflip"] if mirror else []
    if pre_filter:
        vf_parts.append(pre_filter)
    vf_parts.append(f"ass='{ass_arg}'")
    if watermark:
        vf_parts.append(_watermark_filter(watermark))
    vf = ",".join(vf_parts)
    print(f"      [vf] {vf}")

    # Optionally drop the first bg_skip seconds of the background track (skip a
    # long intro / silent lead-in) before it's looped to fill the clip. Reset the
    # timestamps after the trim so aloop and the clip-length atrim below both
    # start from zero.
    bg_pre = (f"atrim=start={bg_skip},asetpts=PTS-STARTPTS,"
              if bg_music and bg_skip and bg_skip > 0 else "")
    if bg_pre:
        print(f"      -> skipping first {bg_skip:g}s of background music")

    try:
        if narration_audio:
            dur = _get_duration(video_path)
            inputs = ["-i", str(video_path), "-i", str(narration_audio)]
            # Movie audio plays at full volume, untouched — the narration is just
            # laid quietly on top (the text band carries the meaning, so the voice
            # doesn't need to duck the film). No ducking = no muted tail.
            # amix with normalize=0 sums inputs rather than averaging them, so the
            # movie keeps its level. narration.wav is already exactly the clip
            # length (tts.py pads it), so no apad is needed — and an *unbounded*
            # apad here would deadlock amix. bg music is looped then hard-trimmed
            # to the clip length for the same reason.
            # Every amix input is first normalized to stereo/48k. Movie files
            # frequently carry 5.1 audio, sometimes with the layout tag lost
            # ("6 channels (unknown)") — amix then produces a stream the AAC
            # encoder can't open ("Error while opening encoder", exit -22).
            norm = "aresample=48000,aformat=channel_layouts=stereo"
            chains = [
                f"[0:a]{norm}[a0]",
                f"[1:a]volume={narration_volume},{norm}[nar]",
            ]
            mix_labels = "[a0][nar]"
            n_mix = 2
            if bg_music:
                inputs += ["-i", str(bg_music)]
                chains.append(
                    f"[2:a]{bg_pre}aloop=loop=-1:size=2000000000,atrim=0:{dur},volume={bg_volume},{norm}[bg]"
                )
                mix_labels += "[bg]"
                n_mix = 3
            # duration=first → the movie ([0:a]) sets the length, so its tail is
            # never clipped; amix pads the finite narration with trailing silence.
            chains.append(
                f"{mix_labels}amix=inputs={n_mix}:duration=first:normalize=0[aout]"
            )
            _run_ffmpeg([
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex", ";".join(chains),
                "-vf", vf,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ])
        elif bg_music and music_duck:
            # Recap Montage mix: the MUSIC is the bed and stays loud; the movie's
            # own dialogue plays underneath and ducks further whenever the music
            # is loud (sidechaincompress keyed off the music). This is the
            # "music-forward, dialogue ducked" look of recap edits — the opposite
            # of the narrated branch, where the movie audio leads.
            dur = _get_duration(video_path)
            norm = "aresample=48000,aformat=channel_layouts=stereo"
            # bg music: skip a lead-in (optional), loop to fill, hard-trim to the
            # clip length (an unbounded pad would deadlock the downstream amix),
            # then split — one copy keys the compressor, one copy is mixed in.
            # Voice-over ducking: the MUSIC ducks under the dialogue, not the
            # other way round. [mvkey] (the lifted dialogue) keys a hard sidechain
            # compressor on the music, so whenever anyone speaks the bed drops far
            # out of the way; in the gaps between dialogue the music returns to its
            # `bg_volume` resting level. The dialogue path [mvout] is never touched.
            # alimiter catches summed peaks so nothing clips.
            # Pad the movie audio to EXACTLY the video length (apad then atrim to
            # `dur`) so the mixed audio always spans the full video — otherwise the
            # body audio can end a touch before the video and it "continues without
            # sound". The padded tail carries no dialogue, so the sidechain doesn't
            # duck the music there and the bed keeps playing to the very end.
            filter_complex = (
                f"[0:a]{norm},{_RECAP_DIALOGUE_NORM},apad,atrim=0:{dur},asetpts=N/SR/TB[mv];"
                f"[mv]asplit=2[mvkey][mvout];"
                f"[1:a]{bg_pre}aloop=loop=-1:size=2000000000,atrim=0:{dur},"
                f"volume={bg_volume},{norm}[bg];"
                f"[bg][mvkey]sidechaincompress=threshold=0.05:ratio=2:attack=50:release=1200[bgduck];"
                f"[mvout][bgduck]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[aout]"
            )
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(bg_music),
                "-filter_complex", filter_complex,
                "-vf", vf,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ])
        elif bg_music:
            dur = _get_duration(video_path)
            # Same stereo/48k normalization as the narrated branch (see above).
            norm = "aresample=48000,aformat=channel_layouts=stereo"
            # normalize=0 SUMS the inputs instead of auto-rescaling by the
            # active-input count: the movie stays at its full level and the bg
            # music sits at exactly bg_volume. The old default (normalize=1 with
            # dropout_transition=2) renormalised over the first 2s, which made the
            # music audibly swell up from quiet to full as the clip began.
            filter_complex = (
                f"[0:a]{norm}[a0];"
                f"[1:a]{bg_pre}aloop=loop=-1:size=2000000000,atrim=0:{dur},"
                f"volume={bg_volume},{norm}[bg];"
                f"[a0][bg]amix=inputs=2:duration=first:normalize=0[aout]"
            )
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(bg_music),
                "-filter_complex", filter_complex,
                "-vf", vf,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ])
        else:
            _run_ffmpeg([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-c:a", "copy",
                str(output_path),
            ])
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}):\n{stderr}") from None
    return output_path


def _run_ffmpeg(cmd: list) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _get_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())
