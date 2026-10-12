"""CLI: turn a movie file + chosen timestamp ranges into a captioned upload with metadata.

An edit is THREE INDEPENDENT CHOICES, not one "mode". Modes used to be an enum,
which meant the only buildable edits were the combinations someone had already
wired up — narrated implied vertical, recap implied no narration, and a narrated storytelling recap in 16:9 was unreachable.

  --selection  who picks the clips
      manual  you passed --ranges
      scene   scenes.py — one self-contained scene
      spine   recap.py — a chronological chain of plot beats
      script  narrative.py — the narration script picks them (see below)

  --voice      what carries the edit, i.e. how the audio is mixed
      none          the film's own dialogue
      commentary    narration.py + TTS, quiet UNDER full dialogue (clip-led)
      storytelling  narrative.py + TTS as the LEAD, dialogue ducked under it
      music         a loud music bed with the dialogue ducked under it

  --shape      vertical | square | landscape — free in every combination

`--mode` still works as a shorthand for a preset (classic / narrated / recap /
storytelling); any axis passed explicitly overrides the preset's value.

The two render orders
---------------------
Every voice except `storytelling` is CLIP-LED: the ranges fix the video length and
the narration is written and spoken to fit inside it. `storytelling` is
NARRATION-LED and runs the other way round — the script is written first, spoken,
and measured, and each clip is then cut to last exactly as long as its line takes
to read. That inversion is why narrative.py could not simply be added as a fourth
enum value.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

from burn import get_watermarks, burn_subtitles, pick_random_audio
from captions import build_ass
from clip_selector import cut_and_concat, parse_ranges, parse_timestamp
from ffutil import get_duration, get_resolution
from outro import (
    load_brand, build_brand_end_card, append_end_card,
    _outro_durations,
    build_animated_outro,
    detect_beats,
    find_iconic_shots,
    snap_cuts_to_beats,
)
from recap import suggest_recap
from scenes import suggest_scenes
from metadata import (
    DEFAULT_EDITING_PROGRAM,
    DEFAULT_MODEL,
    build_edit_description,
    clean_music_name,
    generate_metadata,
)
from narration import NarrationBeat, generate_narration_beats
from narrative import suggest_narrative
from reformat import append_still_image
from silence import detect_silences, keep_intervals, remap_time, snap_ranges_to_speech
from subtitles import (
    cues_for_ranges,
    cues_from_words,
    cues_to_text,
    cues_to_words,
    find_srt_for_video,
    parse_srt,
)
from thumbnail import make_thumbnail
from transcriber import transcribe, words_to_text
from tts import DEFAULT_VOICE, NarrationSegment, synthesize_narration, synthesize_script

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = PROJECT_ROOT / "output"
AUDIO_DIR   = PROJECT_ROOT / "audio"

_JUNK_TOKENS = re.compile(
    r"\b("
    r"1080p|720p|480p|2160p|4k|uhd|imax|"
    r"hdr10\+?|hdr|dolby ?vision|dovi|"
    r"webrip|web[-\s]?dl|webdl|bluray|blu[-\s]?ray|bdrip|brrip|dvdrip|hdtv|remux|"
    r"x264|x265|h ?264|h ?265|hevc|10bit|8bit|"
    # Audio codecs, optionally swallowing a glued channel count (AAC5.1 / DDP5.1
    # arrive here as "aac5 1" after dots became spaces).
    r"(?:aac\d?|ac3|eac3|ddp?\+?\d?|dts(?:[-\s]?hd)?(?:[-\s]?ma)?|truehd)(?:\s[012])?|atmos|"
    r"yts|yify|rarbg|amzn|nf|repack|proper|extended|remastered|directors cut|"
    r"multi|dual audio"
    r")\b",
    re.IGNORECASE,
)
# Audio channel layouts (5.1, 7.1, 2.0) — stripped before dots become spaces.
_CHANNELS_RE = re.compile(r"\b[257][.\s][012]\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Named points in the selection × voice × shape space. These are conveniences for
# the caller, NOT the type system — every other combination stays reachable by
# passing the axes directly.
PRESETS: dict[str, dict[str, str]] = {
    "classic":      {"selection": "manual", "voice": "none",         "shape": "square"},
    "narrated":     {"selection": "manual", "voice": "commentary",   "shape": "vertical"},
    "recap":        {"selection": "spine",  "voice": "music",        "shape": "vertical"},
    "storytelling": {"selection": "script", "voice": "storytelling", "shape": "landscape"},
}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "clip"


def filename_to_title(path: Path) -> str:
    """Best-effort cleanup of a movie/TV filename into a human-readable title."""
    stem = path.stem
    stem = re.sub(r"-[A-Z0-9]{2,}$", "", stem)  # trailing release-group tag, e.g. "-RARBG"
    stem = re.sub(r"[\[\(\{].*?[\]\)\}]", " ", stem)
    stem = _CHANNELS_RE.sub(" ", stem)  # 5.1 / 7.1 / 2.0 before dots turn to spaces
    stem = stem.replace(".", " ").replace("_", " ").replace("-", " ")
    stem = _JUNK_TOKENS.sub(" ", stem)
    stem = _YEAR_RE.sub(" ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem.title() if stem else path.stem


def prompt_video_path() -> Path:
    while True:
        raw = input("Movie file path: ").strip().strip('"')
        path = Path(raw)
        if path.is_file():
            return path
        print(f"  File not found: {path}")


def prompt_ranges() -> list[tuple[float, float]]:
    while True:
        raw = input("Clip range(s) (e.g. '3:20-3:30; 5:15-5:45'): ").strip()
        try:
            return parse_ranges(raw)
        except ValueError as exc:
            print(f"  {exc}")


def check_storytelling_duration(actual, target, shape):
    if shape in ("vertical", "square") and actual > min(target, 180.0) + 0.01:
        raise ValueError(f"Storytelling Short is {actual:.1f}s, exceeding its {min(target, 180.0):.1f}s limit. Shorten/regenerate the script; no video was rendered.")
    if shape == "landscape" and actual > target * 1.15:
        print(f"      ! Narration is {actual:.1f}s versus the {target:g}s target; long-form target is approximate.")


def validate_approved_script(lines):
    """Validate reviewed source windows without changing their order or copy."""
    if not isinstance(lines, list) or not lines:
        raise ValueError("script must be a non-empty list")
    result = []
    previous = -1.0
    for line in lines:
        start, end = float(line["start"]), float(line["end"])
        text = line["narration"]
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0
                or end <= start or start <= previous
                or not isinstance(text, str) or not text.strip()):
            raise ValueError("script needs increasing finite starts, end > start, and non-empty narration")
        result.append({"start": start, "end": end, "range": f"{start:g}-{end:g}", "narration": text})
        previous = start
    return result


def reviewed_ranges_match(actual, expected):
    """Serialized millisecond timestamps tolerate rounding, not editorial changes."""
    return len(actual) == len(expected) and all(
        math.isclose(a, b, rel_tol=0, abs_tol=0.001)
        for cut, approved in zip(actual, expected) for a, b in zip(cut, approved))


def narration_coverage(segments, final_duration, *, byline_only=False):
    """Union of actual spoken timeline intervals, excluding text-only bylines."""
    if byline_only or final_duration <= 0:
        return 0.0
    spoken_seconds, previous_end = 0.0, 0.0
    for start, end in sorted((max(0.0, s.start), min(final_duration, s.end)) for s in segments):
        spoken_seconds += max(0.0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return 100 * spoken_seconds / final_duration


def approved_script_ranges(segs, narration_segments, movie_end, region=None):
    """Speech sets cut length, but reviewed source bounds are hard limits."""
    if len(segs) != len(narration_segments):
        raise ValueError("Approved script lost a narration segment")
    bounds = [0.0] + [s.start for s in narration_segments[1:]] + [narration_segments[-1].end + 0.8]
    ranges = []
    for i, seg in enumerate(segs):
        start = seg["start"]
        end = start + max(1.2, bounds[i + 1] - bounds[i])
        if end > min(seg["end"], movie_end) + 1e-6:
            raise ValueError(f"Approved script segment {i + 1} exceeds its source window; shorten narration or widen the reviewed window")
        if region and not any(start >= a and end <= b for a, b in region):
            raise ValueError(f"Approved script segment {i + 1} falls outside --span")
        ranges.append((start, end))
    return ranges


def _prepare_fingerprint(video_path: Path, ranges: list) -> dict:
    """Identify what a prepared clip_raw.mp4 was actually cut from.

    Written to prepare.json at --prepare-only time and re-checked before any
    --clip-dir reuse. Without this, changing the ranges and re-rendering from
    the same prepared dir silently keeps the OLD cut while the new ranges flow
    on to the caption/narration steps — you get one clip's video carrying
    another clip's dialogue.
    """
    return {
        "video": str(Path(video_path).resolve()),
        "ranges": [[round(float(s), 3), round(float(e), 3)] for s, e in ranges],
    }


class StepPrinter:
    """Prints '[n/N] ...' progress lines (the web UI parses these)."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.n = 0

    def __call__(self, message: str) -> None:
        self.n += 1
        # flush: stdout is a pipe when the web UI spawns us, and Python
        # block-buffers pipes — without this the progress lines arrive in one
        # burst at the end instead of as each step completes.
        print(f"\n[{self.n}/{self.total}] {message}" if self.n == 1 else f"[{self.n}/{self.total}] {message}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a captioned movie short with AI metadata.")
    try:
        brand = load_brand()
        watermarks = get_watermarks()
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"Invalid brand configuration: {exc}")
    parser.add_argument("--end-card", action=argparse.BooleanOptionalAction, default=True,
                        help="Append channel identity and Subscribe card (default: on)")
    parser.add_argument("--end-card-duration", type=float, default=2.0,
                        help="End-card seconds, 1.5–2.5 (default: 2)")
    parser.add_argument("--video", help="Path to the source movie file (skips the prompt)")
    parser.add_argument("--ranges", help="Timestamp ranges, e.g. '3:20-3:30; 5:15-5:45' (skips the prompt)")
    parser.add_argument("--whisper-model", default="medium", help="faster-whisper model size (default: medium)")
    parser.add_argument("--language", default="en", help="Transcription language code (default: en)")
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL, help=f"Ollama model for metadata (default: {DEFAULT_MODEL})")
    parser.add_argument("--source-title", help="Movie/show name override (default: parsed from the filename)")
    parser.add_argument("--title", default=None,
                        help="On-screen title override (default: written by the LLM in the metadata step)")
    parser.add_argument("--editorial-file", help="Approved JSON with metadata and commentary beats or storytelling script source windows")
    parser.add_argument("--size", type=int, default=1080, help="Square-mode side length in pixels (default: 1080)")
    parser.add_argument("--bg-music", default=None, help="Path to background music file (default: random from audio/)")
    parser.add_argument("--bg-volume", type=float, default=0.10, help="Background music volume 0.0–1.0 (default: 0.10)")
    parser.add_argument("--bg-skip", type=float, default=0.0,
                        help="Skip the first N seconds of the background music (e.g. cut a slow intro). Default: 0")
    parser.add_argument("--no-bg-music", action="store_true", help="Disable background music entirely")
    parser.add_argument("--no-loudnorm", action="store_true",
                        help="Skip -14 LUFS loudness mastering on the final audio "
                             "(mastering makes every clip sit at the social-media standard loudness)")
    parser.add_argument(
        "--watermark",
        default="mv-edits",
        choices=list(watermarks.keys()) + ["none"],
        help="Watermark to burn in (bottom centre, fades after 3 s). Use 'none' to disable. (default: mv-edits)",
    )
    # ── narrated-format options ──────────────────────────────────────────────
    parser.add_argument("--narrate", action="store_true",
                        help="Narrated vertical format: LLM scene narration read by TTS + text beneath the video")
    parser.add_argument("--audio-mix", choices=["original", "narrator-led", "narration-only"], default=None,
                        help="Narration mix, independent of selection; omitted preserves existing mode defaults")
    parser.add_argument("--narration-volume", type=float, default=0.04,
                        help="Narration voiceover volume 0.0–1.0 (default: 0.04)")
    parser.add_argument("--tts-voice", default=DEFAULT_VOICE,
                        help=f"edge-tts voice for the narration (default: {DEFAULT_VOICE})")
    parser.add_argument("--srt", default=None,
                        help="Subtitle file for captions (default: auto-detected next to the movie; falls back to Whisper)")
    parser.add_argument("--no-srt", action="store_true",
                        help="Ignore any subtitle file and transcribe with Whisper instead")
    parser.add_argument("--layout", choices=["square", "vertical", "landscape"], default=None,
                        help="Frame layout (default: vertical when --narrate, else square)")
    parser.add_argument("--clip-dir", default=None,
                        help="Use this exact output dir (reuses its clip_raw.mp4 / thumbnail.jpg if present)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only cut the raw clip into --clip-dir, then exit (for the two-phase thumbnail flow)")
    parser.add_argument("--thumbnail", action="store_true",
                        help="Always compose a thumbnail, including for classic edits that skip it by default")
    parser.add_argument("--no-thumbnail", action="store_true",
                        help="Skip composing a thumbnail entirely (also disables baking one into the video)")
    parser.add_argument("--thumbnail-size", default=None,
                        help="Thumbnail canvas as WxH (default: follows --layout — 1080x1920 vertical, "
                             "1920x1080 landscape, square for square)")
    parser.add_argument("--prepend-thumbnail", action="store_true",
                        help="Bake the thumbnail as the last 0.1s of the final video (pick it as the Short's thumbnail on mobile)")
    parser.add_argument("--remove-silence", action="store_true",
                        help="Cut sections where the audio is near-silent (no dialogue) before rendering (default: off)")
    parser.add_argument("--caption-font", default="Arial Black",
                        help="Caption font family (must be installed on the system; default: Arial Black)")
    parser.add_argument("--caption-animation", choices=["karaoke", "fade", "pop"], default="karaoke",
                        help="Caption effect: karaoke (word-by-word), fade (whole line), pop (scale-in)")
    parser.add_argument("--caption-color", default="white",
                        help="Fixed caption colour when shuffle is off (white/yellow/green/cyan/pink/orange)")
    parser.add_argument("--no-caption-shuffle", action="store_true",
                        help="Use one fixed --caption-color instead of cycling colours per sentence")
    parser.add_argument("--caption-font-shuffle", action="store_true",
                        help="Cycle a different caption font per sentence, starting from --caption-font "
                             "(works from an .srt or from Whisper — both feed the same word list)")
    parser.add_argument("--caption-italic", action="store_true",
                        help="Render the burned captions in italics")
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror (flip left-right) the clip footage only — captions/narration text stay upright")
    # ── the three axes ───────────────────────────────────────────────────────
    parser.add_argument("--mode", choices=sorted(PRESETS), default=None,
                        help="Preset: a named point in the selection/voice/shape space. "
                             "Any axis passed explicitly wins over the preset.")
    parser.add_argument("--selection", choices=["manual", "scene", "spine", "script"], default=None,
                        help="Who picks the clips (default: from --mode)")
    parser.add_argument("--voice", choices=["none", "commentary", "storytelling", "music"], default=None,
                        help="What carries the edit / how the audio is mixed (default: from --mode)")
    parser.add_argument("--shape", choices=["square", "vertical", "landscape"], default=None,
                        help="Frame shape — an alias of --layout, free in every combination")
    parser.add_argument("--dialogue-volume", type=float, default=0.22,
                        help="Movie dialogue level under a storytelling narrator (default: 0.22)")
    parser.add_argument("--no-dialogue-captions", action="store_true",
                        help="Don't burn the film's dialogue as captions (default ON for storytelling, "
                             "whose narration renders as a lower third instead)")
    parser.add_argument("--byline-only", action="store_true", help="Reviewed commentary as brief text only; preserve movie audio")
    # ── recap / montage options ──────────────────────────────────────────────
    parser.add_argument("--outro-seconds", type=float, default=5.0,
                        help="Recap mode: length of the beat-synced iconic outro montage (default: 5)")
    parser.add_argument("--outro-shots", default=None,
                        help="Recap mode: comma-separated source timestamps (e.g. '1:02:10, 1:15:03') forced as "
                             "outro shots, merged with the auto face-scan")
    parser.add_argument("--auto-outro-shots", type=int, default=12,
                        help="Recap mode: how many iconic face shots to auto-find for the outro (0 disables the scan)")
    parser.add_argument("--no-outro", action="store_true",
                        help="Recap mode: skip the iconic outro montage")
    parser.add_argument("--target-seconds", type=float, default=120.0,
                        help="Recap mode: total body length to aim for when auto-building ranges (default: 120)")
    parser.add_argument("--recap-model", default=None,
                        help="Recap mode: Ollama model for auto-building ranges from the whole subtitle file "
                             "(default: a local model — the whole SRT is fed to it)")
    parser.add_argument("--no-snap-speech", action="store_true",
                        help="Skip snapping clip boundaries to speech gaps (faster, but lines "
                             "may cut mid-sentence where the subtitle timing drifts from the audio)")
    parser.add_argument("--span", default=None,
                        help="Only summarize this stretch of the movie, e.g. '30:00-45:00'. The auto "
                             "clip-picker (spine/script) reads and picks from just this span.")
    parser.add_argument("--hook", action="store_true",
                        help="Prepend a cold-open HOOK: the 1-3 punchiest short lines as a teaser at the start")
    parser.add_argument("--hook-count", type=int, default=3,
                        help="How many hook lines to use when --hook is on (default: 3)")
    # ── description credits (YouTube "edit" layout) ──────────────────────────
    parser.add_argument("--edit-credits", action="store_true",
                        help="Write the description in the edit layout: hook + Movie/Music/Editing Program "
                             "credits + hashtags + fair-use disclaimer (default ON for recap)")
    parser.add_argument("--no-edit-credits", action="store_true",
                        help="Force the plain description even in recap mode")
    parser.add_argument("--music-credit", default=None,
                        help="Music track name for the 'Music:' credit line (default: derived from --bg-music filename)")
    parser.add_argument("--editing-program", default=DEFAULT_EDITING_PROGRAM,
                        help=f"'Editing Program:' credit line (default: {DEFAULT_EDITING_PROGRAM!r})")
    parser.add_argument("--no-disclaimer", action="store_true",
                        help="Omit the fair-use copyright disclaimer from the edit-style description")
    args = parser.parse_args()
    if not math.isfinite(args.end_card_duration) or not 1.5 <= args.end_card_duration <= 2.5:
        parser.error("--end-card-duration must be between 1.5 and 2.5 seconds")

    video_arg = args.video.strip().strip('"').strip("'") if args.video else None
    video_path = Path(video_arg) if video_arg else prompt_video_path()
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    ranges = parse_ranges(args.ranges) if args.ranges else []
    # A movie SPAN scopes the auto clip-picker to one stretch of the film so the
    # user can summarize sections in separate renders.
    span_region = parse_ranges(args.span) if args.span else None
    source_title = args.source_title or filename_to_title(video_path)

    # ── Axis resolution ──────────────────────────────────────────────────────
    # A preset supplies defaults for all three axes; anything passed explicitly
    # wins. Nothing here may force one axis from another — that coupling is the
    # bug this replaced.
    preset = PRESETS[args.mode] if args.mode else dict(PRESETS["classic"])
    if args.mode is None and args.narrate:
        preset = dict(PRESETS["narrated"])          # legacy --narrate
    selection = args.selection or preset["selection"]
    voice     = args.voice     or preset["voice"]
    # --layout is the old name for --shape; either may be given.
    layout_mode = args.shape or args.layout or preset["shape"]

    # Derived facts about this combination. Everything downstream branches on
    # these, never on a mode name.
    narration_led = voice == "storytelling"     # script decides the clip lengths
    # Any edit carrying a narrator leads with it by default. The old default put
    # commentary 25 dB UNDER the film — measurably inaudible, which is what makes
    # an edit look like a re-upload with a voice faintly over it rather than a
    # piece of commentary. `--audio-mix original` still asks for the old balance.
    audio_mix = args.audio_mix or (
        "narrator-led" if voice in ("commentary", "storytelling") else "original")
    if args.audio_mix and voice not in ("commentary", "storytelling"):
        parser.error("--audio-mix requires commentary or storytelling voice")
    editorial = None
    if args.editorial_file:
        try:
            editorial = json.loads(Path(args.editorial_file).read_text(encoding="utf-8"))
            if voice not in ("commentary", "storytelling"):
                raise ValueError("--editorial-file requires commentary or storytelling voice")
            if not isinstance(editorial, dict) or not all(isinstance(editorial.get(k), str) and editorial[k].strip() for k in ("title", "description")):
                raise ValueError("editorial title and description must be non-empty strings")
            if not isinstance(editorial.get("tags"), list) or not all(isinstance(t, str) for t in editorial["tags"]):
                raise ValueError("editorial tags must be a list of strings")
            if voice == "storytelling":
                editorial["script"] = validate_approved_script(editorial.get("script"))
            else:
                lines = editorial.get("commentary")
                if not isinstance(lines, list) or not lines:
                    raise ValueError("editorial commentary must be a non-empty list")
                previous = -1.0
                for line in lines:
                    anchor = float(line["anchor"])
                    if not math.isfinite(anchor) or anchor < 0 or anchor <= previous or not isinstance(line["text"], str) or not line["text"].strip():
                        raise ValueError("commentary anchors must increase and text must be non-empty")
                    previous = anchor
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(f"Invalid editorial file: {exc}")
    auto_ranges   = selection in ("spine", "script")
    if args.byline_only and (voice != "commentary" or not editorial):
        parser.error("Text-only bylines require reviewed commentary")
    if editorial and editorial.get("comparison_style"):
        expected = [(float(s), float(e)) for s, e in editorial["source_ranges"]]
        if video_path.resolve() != Path(editorial["source_video"]).resolve() or not reviewed_ranges_match(ranges, expected):
            parser.error("Comparison source/ranges changed; regenerate the review plan")
        if args.hook or args.remove_silence or not args.no_snap_speech:
            parser.error("Comparison timing is fixed; disable hook, silence removal and speech snapping")
    args.narrate  = voice == "commentary"       # the legacy flag == clip-led narration
    music_leads   = voice == "music"
    # Dialogue captions default off under a storytelling narrator — its lines
    # render as a lower third instead, and two text layers would collide.
    dialogue_captions = not (args.no_dialogue_captions or narration_led)

    print(f"Edit: selection={selection} voice={voice} shape={layout_mode}"
          + (f"  (preset: {args.mode})" if args.mode else ""))

    # A selection agent builds its own ranges further down, so only the manual
    # paths need them up front (prompt when interactive).
    if narration_led and selection != "script":
        parser.error("Storytelling voice requires --selection script; manual/scene/spine ranges cannot be silently replaced.")
    if selection == "script" and not narration_led:
        parser.error("Script selection requires --voice storytelling.")
    if narration_led and ranges:
        parser.error("Storytelling derives ranges from its script; use --span to scope the movie instead of --ranges.")
    if not ranges and selection == "manual":
        ranges = prompt_ranges()

    # Release year (from the filename) → shown after the movie name on the thumbnail.
    year_match = _YEAR_RE.search(video_path.stem)
    movie_label = f"{source_title} ({year_match.group(0)})" if year_match else source_title

    # Subtitle file: explicit flag, else auto-detect next to the movie (--no-srt disables)
    srt_path: Path | None = None
    if args.no_srt:
        pass
    elif args.srt:
        srt_path = Path(args.srt)
        if not srt_path.is_file():
            print(f"Error: SRT file not found: {srt_path}")
            sys.exit(1)
    else:
        srt_path = find_srt_for_video(video_path)

    # Any selection agent reads the whole subtitle file, so it is a hard
    # requirement for spine and script — but for nothing else.
    if selection == "scene" and not ranges and srt_path is None:
        print("Error: --selection scene needs a subtitle file (--srt) or explicit --ranges.")
        sys.exit(1)
    if auto_ranges and srt_path is None:
        print(f"Error: --selection {selection} needs a subtitle file (--srt) — it reads the whole film.")
        sys.exit(1)

    # Music is resolved up front (not at mix time) whenever the cuts might be
    # snapped to its beats, so the same track drives the analysis and the mix.
    recap_music: Path | None = None
    if music_leads and not args.no_bg_music:
        if args.bg_music:
            recap_music = Path(args.bg_music)
            if not recap_music.is_file():
                print(f"Error: music track not found: {recap_music}")
                sys.exit(1)
        else:
            # "Random from audio/" — resolve it now, or phase 2 rolls a different
            # track and snaps to a different beat grid.
            recap_music = pick_random_audio(AUDIO_DIR)
            if recap_music is not None:
                args.bg_music = str(recap_music)
                print(f"Music bed (random): {recap_music.name}")
    # Snapping only makes sense when the music is what the edit is cut to. Under a
    # narrator, nudging cuts onto a beat grid would fight the narration's timing.
    snap_to_beats = music_leads and recap_music is not None

    if args.clip_dir:
        # Phase 2 (or reuse): an existing dir prepared earlier — keep its clip_raw
        # and any hand-picked thumbnail.jpg.
        clip_dir = Path(args.clip_dir)
        clip_dir.mkdir(parents=True, exist_ok=True)
    else:
        slug = slugify(video_path.stem)[:10]
        clip_dir = OUTPUT_DIR / slug
        if clip_dir.exists():
            n = 2
            while (OUTPUT_DIR / f"{slug}-{n}").exists():
                n += 1
            clip_dir = OUTPUT_DIR / f"{slug}-{n}"
        clip_dir.mkdir(parents=True, exist_ok=True)

    raw_clip_path   = clip_dir / "clip_raw.mp4"
    ass_path        = clip_dir / "captions.ass"
    final_path      = clip_dir / "clip_final.mp4"
    metadata_path   = clip_dir / "metadata.json"
    description_path = clip_dir / "description.txt"
    narration_txt   = clip_dir / "narration.txt"
    narration_wav   = clip_dir / "narration.wav"
    thumbnail_path  = clip_dir / "thumbnail.jpg"

    print(f"Source title detected: {source_title!r}")
    if srt_path:
        print(f"Subtitle file: {srt_path}")

    # Phase 1: just cut the raw clip so the web UI can pick a thumbnail from it.
    if args.prepare_only:
        print("\n[prepare] Cutting raw clip...")
        # A recap render cuts the BEAT-SNAPPED ranges, not the raw ones, so the
        # prepared clip has to be snapped too — otherwise phase 2 sees a cut it
        # cannot account for, re-cuts, and throws away the hand-picked thumbnail.
        # The fingerprint stays on the RAW ranges: that is what phase 2 compares
        # against, and snapping is deterministic for the same track.
        prep_ranges = ranges
        if snap_to_beats:
            prep_ranges = snap_cuts_to_beats(ranges, detect_beats(recap_music))
            print(f"      -> snapped {len(prep_ranges)} beat(s) to {recap_music.name}")
        # Speech-gap snap so the PREPARED cut doesn't chop dialogue mid-line (the
        # intro cutoff). Phase 1 owns the cut, so it must snap here; phase 2 snaps
        # the caption ranges the same deterministic way, so they stay aligned.
        if voice != "storytelling" and not args.no_snap_speech:
            prep_ranges = snap_ranges_to_speech(video_path, prep_ranges)
            print(f"      -> snapped {len(prep_ranges)} clip boundaries to speech gaps")
        # Fade (dissolve) between clips whenever there's more than one, in every
        # mode — not just the auto-picked ones.
        cut_and_concat(video_path, prep_ranges, raw_clip_path,
                       fade_edges=(auto_ranges or len(prep_ranges) > 1))
        (clip_dir / "prepare.json").write_text(
            json.dumps(_prepare_fingerprint(video_path, ranges), indent=2), encoding="utf-8")
        print("\nPrepared.")
        if recap_music is not None:
            print(f"  Recap music:  {recap_music}")
        print(f"  Clip dir:     {clip_dir}")
        print(f"  Raw clip:     {raw_clip_path}")
        print(f"  Movie:        {movie_label}")
        return

    # 5 base steps: cut, captions, metadata, build-ass, burn. Framing (square /
    # vertical canvas) is folded into the burn encode, not a step of its own.
    # A thumbnail is composed whenever the narrated format needs one OR it's
    # being baked into the video (--prepend-thumbnail), so every pick-first run —
    # classic included — accounts for the compose + bake steps.
    if args.thumbnail_size:
        try:
            _tw, _th = (int(v) for v in str(args.thumbnail_size).lower().split("x", 1))
            thumb_size = (max(64, _tw), max(64, _th))
        except (ValueError, TypeError):
            print(f"      ! unparseable --thumbnail-size {args.thumbnail_size!r}, using the layout default")
            thumb_size = None
    else:
        thumb_size = None
    if thumb_size is None:
        thumb_size = {
            "vertical":  (1080, 1920),
            "square":    (args.size, args.size),
            "landscape": (1920, 1080),
        }[layout_mode]

    if args.no_thumbnail:
        # An explicit opt-out also cancels the bake-in — there is nothing to bake.
        args.prepend_thumbnail = False
    spine = selection == "spine"
    # --thumbnail asks for one outright. Without it the old implicit rule stands
    # (narrated/agent-picked edits get one, classic doesn't), which is why a
    # classic upload used to come out with no thumbnail to choose on YouTube.
    want_thumbnail = ((args.thumbnail or args.narrate or args.prepend_thumbnail or auto_ranges)
                      and not args.no_thumbnail)
    total_steps = (5
                   + (2 if spine else 0)                   # spine: beat/shot analysis + animated outro
                   - (1 if (spine and args.no_outro) else 0)     # ...minus the outro when it is off
                   + (2 if narration_led else 0)           # write the script + speak it
                   + (1 if (selection == "scene" and not ranges) else 0)  # pick the scene
                   + (1 if (args.remove_silence and not auto_ranges) else 0)
                   + (2 if args.narrate else 0)            # narration beats + voiceover
                   + (1 if want_thumbnail else 0)          # compose/use thumbnail
                   + (1 if args.prepend_thumbnail else 0)) # bake thumbnail into video
    step = StepPrinter(total_steps)

    # Recap analysis: beat-track the music, snap the body clips onto the beat
    # grid, and build the beat-synced iconic outro. cut_ranges (body + outro) is
    # what gets cut; caption_ranges (body only) is what gets captioned — the
    # outro flashes carry no dialogue.
    cut_ranges = ranges
    caption_ranges = ranges
    recap_shot_times: list[float] = []
    recap_shot_faces: list = []
    recap_outro_durations: list[float] = []
    recap_body_duration = 0.0
    narration_segments: list = []

    # ── NARRATION-LED path ───────────────────────────────────────────────────
    # The only place in this file where the script runs before the cut. The
    # narrator's lines are written, spoken and measured first, and each clip is
    # then cut to last exactly as long as its line takes to read.
    if narration_led:
        script_model = args.recap_model or "gemma4:e4b"
        step("Loading approved storytelling script..." if editorial else f"Writing the narration script from subtitles ({script_model})...")
        # A feature-length script is several model calls, so report each one —
        # otherwise the step sits silent for minutes and looks hung.
        try:
            segs = editorial["script"] if editorial else suggest_narrative(srt_path, source_title=source_title, model=script_model,
                                     target_seconds=args.target_seconds, region=span_region,
                                     progress=lambda msg: print(f"      -> {msg}", flush=True))
        except ValueError as exc:
            print(f"Error: {exc}")
            print("       Try a bigger model, or a shorter target length (fewer segments per call).")
            sys.exit(1)
        span = f"{segs[0]['range']}-{segs[-1]['range']}"
        print(f"      -> {len(segs)} script segments covering {span}")

        # Persist the script BEFORE speaking it. Generating a feature-length
        # script is minutes of model calls; speech is a network service that
        # fails intermittently. Writing this afterwards meant one transient
        # rejection discarded every generated segment with nothing to resume
        # from. The file doubles as the starting point for an --editorial-file.
        script_json = clip_dir / "script.json"
        script_json.write_text(json.dumps(segs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"      -> script saved to {script_json.name} (survives a speech failure)")

        step(f"Speaking the script (edge-tts, {args.tts_voice}) — its length sets each clip's length...")
        try:
            narration_segments = synthesize_script([s["narration"] for s in segs],
                                                   narration_wav, voice=args.tts_voice)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            print(f"       The script survived at {script_json} — rerun with "
                  f"--editorial-file to speak it again without regenerating.")
            sys.exit(1)
        narration_txt.write_text(
            "\n".join(f"[{s.start:07.1f}s] {s.text}" for s in narration_segments),
            encoding="utf-8")

        # Clip i runs from where its line starts to where the next one starts, so
        # the concatenated footage and the voice track share one timeline.
        measured_seconds = narration_segments[-1].end + 0.8
        try:
            check_storytelling_duration(measured_seconds + (args.end_card_duration if args.end_card else 0), args.target_seconds, layout_mode)
        except ValueError as exc:
            parser.error(str(exc))
        movie_end = get_duration(video_path)
        starts = [s.start for s in narration_segments]
        bounds = [0.0] + starts[1:] + [narration_segments[-1].end + 0.8]
        if editorial:
            try:
                ranges = approved_script_ranges(segs, narration_segments, movie_end, span_region)
            except ValueError as exc:
                parser.error(str(exc))
        else:
            built: list[tuple[float, float]] = []
            for i, seg_meta in enumerate(segs[:len(narration_segments)]):
                span = max(1.2, bounds[i + 1] - bounds[i])
                src_start = float(seg_meta["start"])
                built.append((src_start, min(src_start + span, movie_end)))
            ranges = [(s, e) for s, e in built if e > s + 0.4]
        if not ranges:
            print("Error: the script produced no usable clip ranges.")
            sys.exit(1)
        cut_ranges = ranges
        caption_ranges = ranges
        spoken = narration_segments[-1].end
        print(f"      -> {len(ranges)} clips derived from {spoken:.0f}s of narration "
              f"(~{spoken / 60:.1f} min)")

    if selection == "scene" and not ranges:
        scene_model = args.recap_model or "gemma4:e4b"
        step(f"Picking the best standalone scene from the subtitles ({scene_model})...")
        picks = suggest_scenes(srt_path, source_title=source_title, model=scene_model, count=1,
                               min_seconds=max(8.0, args.target_seconds * 0.7),
                               max_seconds=max(20.0, args.target_seconds * 1.4))
        if not picks:
            print("Error: no scene came back. Pass --ranges, or try a different model.")
            sys.exit(1)
        ranges = [(picks[0]["start"], picks[0]["end"])]
        cut_ranges = caption_ranges = ranges
        print(f"      -> {picks[0].get('title') or 'scene'}: {picks[0]['range']}")

    if spine:
        step("Building recap from subtitles"
             + (" + analyzing music beats" if recap_music is not None else "")
             + " + iconic shots...")
        # How recap reads the provided ranges:
        #   - SHORT ones (<=30s) are literal beats — a hand-picked beat or the
        #     Build-recap preview the user tweaked → use them as-is.
        #   - a LONG one (>30s) is a TIME REGION to scope the recap to ("recap
        #     this stretch of the movie") → auto-pick short beats within it.
        #   - none → auto-pick from the whole movie.
        # This also prevents a long span (e.g. "00:00-40:00") from being cut as a
        # single 40-minute beat, which cut+captioned+burned a huge vertical and
        # looked hung. The LLM feed is always a LOCAL model by default — the whole
        # (or scoped) SRT goes into the prompt and the user is bandwidth-sensitive.
        _f = lambda t: f"{int(t // 60)}:{int(t % 60):02d}"
        literal = [(s, e) for s, e in ranges if e - s <= 30.0]
        # An explicit --span wins; else a long (>30s) range is treated as the region.
        regions = span_region or [(s, e) for s, e in ranges if e - s > 30.0]
        if literal and not regions:
            ranges = literal
        else:
            recap_model = args.recap_model or "gemma4:e4b"
            if regions:
                print("      -> scoping recap to region(s) "
                      + ", ".join(f"{_f(s)}-{_f(e)}" for s, e in regions)
                      + f", picking beats via {recap_model}...")
            else:
                print(f"      -> no clip ranges — building recap from the whole subtitle file ({recap_model})...")
            built = suggest_recap(srt_path, source_title=source_title, model=recap_model,
                                  target_seconds=args.target_seconds, region=regions or None)
            ranges = [(b["start"], b["end"]) for b in built]
            if not ranges:
                print("Error: recap produced no beats. Try the Build-recap button, or pass short --ranges.")
                sys.exit(1)
            print(f"      -> built {len(ranges)} beats (~{sum(e - s for s, e in ranges):.0f}s)")
        if snap_to_beats:
            beats = detect_beats(recap_music)
            snapped_body = snap_cuts_to_beats(ranges, beats)
        else:
            # No music bed — nothing to sync to, so the beats recap.py picked are
            # already the cut. They stay on their complete-sentence boundaries.
            beats = []
            snapped_body = list(ranges)
        # Snap every cut to a real speech GAP so lines aren't chopped mid-word or
        # bled in from the next scene (subtitle timings drift from the audio).
        if not args.no_snap_speech:
            print("      Verifying clip boundaries against the audio (silence snap)...", flush=True)
            snapped_body = snap_ranges_to_speech(video_path, snapped_body)
            print(f"      -> snapped {len(snapped_body)} clip boundaries to speech gaps")
        caption_ranges = snapped_body
        video_duration = get_duration(video_path)

        recap_body_duration = sum(e - s for s, e in snapped_body)
        cut_ranges = snapped_body  # the outro is a separate ANIMATED segment, not cut ranges

        shot_times: list[float] = []
        shot_faces: list = []   # parallel face-centre (0..1) per shot, when known
        if args.outro_shots:  # explicit hand-picked timestamps win
            for tok in re.split(r"[;,]", args.outro_shots):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    shot_times.append(parse_timestamp(tok))
                except ValueError:
                    print(f"      ! ignoring unparseable outro timestamp {tok!r}")
        if not args.no_outro and not shot_times and snapped_body:
            # Outro clips are high-confidence FACE shots found WITHIN the recap's
            # own story beats — so they're both proper close-up faces and from the
            # moments just shown. Fall back to the beats' centres if too few faces.
            n = max(2, min(6, args.auto_outro_shots or 5, len(snapped_body)))
            faces: list = []
            try:
                faces = find_iconic_shots(video_path, count=n * 3, windows=snapped_body,
                                          min_score=0.04, min_gap=1.5)
                if len(faces) < n:  # too few in-story faces — top up from the back half
                    more = find_iconic_shots(video_path, count=n * 3, region=(0.35, 0.98),
                                             min_score=0.04, min_gap=1.5)
                    for m in more:
                        if all(abs(m["time"] - f["time"]) >= 1.5 for f in faces):
                            faces.append(m)
            except Exception as exc:
                print(f"      ! outro face-scan failed ({exc})")
            if faces:
                # ALL outro shots must be real faces — keep the strongest n, in
                # chronological order, and carry each face's centre for framing.
                best = sorted(sorted(faces, key=lambda f: -f["score"])[:n], key=lambda f: f["time"])
                shot_times = [f["time"] for f in best]
                shot_faces = [f.get("fx") for f in best]
                print(f"      -> outro: {len(shot_times)} high-confidence face shot(s)")
            else:
                idxs = sorted({round(k * (len(snapped_body) - 1) / max(1, n - 1)) for k in range(n)})
                shot_times = [(snapped_body[i][0] + snapped_body[i][1]) / 2 for i in idxs]
                shot_faces = [None] * len(shot_times)
                print("      -> outro: no confident faces found; using beat centres")

        if not args.no_outro and shot_times:
            recap_shot_times = shot_times
            recap_shot_faces = shot_faces if len(shot_faces) == len(shot_times) else []
            recap_outro_durations = _outro_durations(
                beats, recap_body_duration, args.outro_seconds, len(recap_shot_times))
        sync_note = (f"{len(beats)} beats | {len(snapped_body)} body clips snapped"
                     if recap_music is not None
                     else f"no music track — {len(snapped_body)} body clips unsnapped")
        print(f"      -> {sync_note} "
              f"(~{recap_body_duration:.0f}s) | {len(recap_shot_times)} outro slow-mo clip(s)")

    # Speech-gap snap for the NON-spine paths (spine already snapped before its
    # outro face-scan). Only where the film's dialogue is actually heard — under a
    # storytelling narrator the movie audio is muted, so there's nothing to protect.
    if not spine and not args.no_snap_speech and cut_ranges and voice != "storytelling":
        print("      Verifying clip boundaries against the audio (silence snap)...", flush=True)
        cut_ranges = snap_ranges_to_speech(video_path, cut_ranges)
        caption_ranges = cut_ranges
        print(f"      -> snapped {len(cut_ranges)} clip boundaries to speech gaps")

    # Optional cold-open HOOK: prepend the punchiest short lines as a teaser. Not
    # for storytelling yet (that wants a narrator teaser line, not raw dialogue).
    # Say so instead of accepting the flag and doing nothing with it — a silently
    # ignored flag reads as a working feature in the render log.
    if args.hook and narration_led:
        print("      ! --hook is not supported for storytelling (narration-led) edits yet "
              "and was ignored. The script's own opening line is the hook.", flush=True)
    if args.hook and srt_path and not narration_led:
        hook_model = args.recap_model or "gemma4:e4b"
        print(f"      Building cold-open hook ({hook_model})...", flush=True)
        try:
            from hook import suggest_hook
            # A single-scene teaser must promise the scene we actually show.
            hook_region = cut_ranges if selection == "scene" else span_region
            hooks = suggest_hook(srt_path, source_title=source_title, model=hook_model,
                                 count=max(1, args.hook_count), region=hook_region)
        except Exception as exc:
            print(f"      ! hook failed ({exc}); skipping")
            hooks = []
        if hooks:
            hook_ranges = [(h["start"], h["end"]) for h in hooks]
            cut_ranges = hook_ranges + cut_ranges
            caption_ranges = hook_ranges + caption_ranges
            recap_body_duration += sum(e - s for s, e in hook_ranges)  # keep the outro's music aligned
            print(f"      -> prepended {len(hook_ranges)} hook clip(s): "
                  + "; ".join(h["range"] for h in hooks))

    # Only reuse a prepared cut when it was cut from THIS video and THESE
    # ranges — otherwise re-cut. A stale reuse renders the old clip's video
    # while the new ranges drive captions/narration, so the dialogue belongs to
    # a different clip entirely.
    reuse_ok = False
    if raw_clip_path.is_file() and args.clip_dir:
        want = _prepare_fingerprint(video_path, ranges)
        try:
            have = json.loads((clip_dir / "prepare.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            have = None
        reuse_ok = have == want
        if not reuse_ok:
            why = "cut from different ranges/video" if have else "no prepare.json (cut by an older version)"
            print(f"  ! Prepared clip does not match this request ({why}) — re-cutting.")
            # Any hand-picked thumbnail in this dir was picked off the OLD cut,
            # so it shows a frame that isn't in the new clip. Drop it and let
            # the thumbnail step compose a fresh one.
            if thumbnail_path.is_file():
                thumbnail_path.unlink()
                print("  ! Discarded the thumbnail picked from the old cut.")

    # Approved commentary is checked against the cut it will sit on — but the
    # anchors and the planned length are both known NOW, so check what we can
    # before spending an encode. Silence removal only shortens the clip, so this
    # estimate is optimistic and can never reject a line that would have fitted.
    if editorial and voice == "commentary" and cut_ranges:
        planned = sum(e - s for s, e in cut_ranges)
        last = float(editorial["commentary"][-1]["anchor"])
        if last >= planned:
            parser.error(
                f"Commentary anchor at {last:g}s falls outside the {planned:.1f}s of footage "
                f"these ranges produce. Move the anchor or widen the ranges; nothing was rendered.")

    if reuse_ok:
        step(f"Reusing prepared raw clip ({raw_clip_path.name})...")
        print(f"      -> {raw_clip_path}")
    else:
        step(f"Cutting {len(cut_ranges)} range(s) from source video...")
        cut_and_concat(video_path, cut_ranges, raw_clip_path,
                       fade_edges=(auto_ranges or len(cut_ranges) > 1))
        print(f"      -> {raw_clip_path}")
    clip_duration = get_duration(raw_clip_path)

    # Optional silence cut: remove spans where the audio is near-zero (no
    # dialogue). Subtitle cue times are remapped through silence_keeps below;
    # the Whisper path needs no remap because it transcribes the trimmed clip.
    framing_src = raw_clip_path
    silence_keeps: list[tuple[float, float]] | None = None
    if args.remove_silence and not auto_ranges:
        step("Removing blank spaces (near-silent gaps)...")
        silences = detect_silences(raw_clip_path)
        keeps = keep_intervals(clip_duration, silences)
        removed = clip_duration - sum(e - s for s, e in keeps)
        if keeps and removed > 0.3:
            trimmed_path = clip_dir / "clip_trimmed.mp4"
            cut_and_concat(raw_clip_path, keeps, trimmed_path)
            silence_keeps = keeps
            framing_src = trimmed_path
            clip_duration = get_duration(trimmed_path)
            print(f"      -> cut {removed:.1f}s across {len(silences)} gap(s), "
                  f"clip is now {clip_duration:.1f}s -> {trimmed_path.name}")
        else:
            print("      -> no silent gaps worth removing")

    # Framing is a filter prefix applied inside the burn encode (one x264
    # generation instead of two). The vertical layout geometry is fixed
    # arithmetic, so no probe/encode is needed to know it.
    layout: dict | None = None
    if layout_mode == "vertical":
        frame_w, frame_h = 1080, 1920
        video_top = (frame_h - frame_w) // 2
        layout = {"width": frame_w, "height": frame_h,
                  "video_top": video_top, "video_bottom": video_top + frame_w}
        pre_filter = (f"crop=min(iw\\,ih):min(iw\\,ih),"
                      f"scale={frame_w}:{frame_w}:flags=lanczos,"
                      f"pad={frame_w}:{frame_h}:0:{video_top}:black")
        frame_size = (frame_w, frame_h)
    elif layout_mode == "landscape":
        # 16:9 1080p, no crop — keep the original framing, letterbox if the source
        # isn't 16:9. This is the long-form / narrative shape.
        fw, fh = 1920, 1080
        pre_filter = (f"scale={fw}:{fh}:force_original_aspect_ratio=decrease:flags=lanczos,"
                      f"pad={fw}:{fh}:(ow-iw)/2:(oh-ih)/2:black")
        frame_size = (fw, fh)
    else:
        pre_filter = (f"crop=min(iw\\,ih):min(iw\\,ih),"
                      f"scale={args.size}:{args.size}:flags=lanczos")
        frame_size = (args.size, args.size)

    if srt_path:
        step("Extracting dialogue captions from subtitle file...")
        cues = cues_for_ranges(parse_srt(srt_path), caption_ranges)
        if silence_keeps:
            # Shift cue times onto the silence-trimmed timeline (cues sit in
            # the kept spans by definition — silence has no dialogue).
            for cue in cues:
                cue.start = remap_time(cue.start, silence_keeps)
                cue.end = max(remap_time(cue.end, silence_keeps), cue.start + 0.05)
        words = cues_to_words(cues)
        transcript = cues_to_text(cues)
        print(f"      -> {len(cues)} cues / {len(words)} words from {srt_path.name}")
    else:
        step("Transcribing clip with faster-whisper (no subtitle file found)...")
        words = transcribe(framing_src, model_size=args.whisper_model, language=args.language, source_title=source_title)
        transcript = words_to_text(words)
        cues = cues_from_words(words)
        print(f"      -> {len(words)} words transcribed")

    # (A storytelling narration was already written, spoken and measured above —
    # it had to be, because it decided where the cuts went.)
    if args.narrate:
        step("Loading approved timed commentary..." if editorial else f"Writing timed scene narration via Ollama ({args.ollama_model})...")
        # Surrounding dialogue from the full subtitle file — the lead-in usually
        # names who is present, so the narrator stops guessing the speaker.
        scene_before = scene_after = ""
        if srt_path:
            all_cues = parse_srt(srt_path)
            first_start, last_end = ranges[0][0], ranges[-1][1]
            before = [c.text for c in all_cues if first_start - 120 <= c.end <= first_start]
            after = [c.text for c in all_cues if last_end <= c.start <= last_end + 30]
            scene_before = " ".join(" ".join(before).split()[-180:])
            scene_after = " ".join(" ".join(after).split()[:60])
        if editorial:
            lines = editorial["commentary"]
            if float(lines[-1]["anchor"]) >= clip_duration:
                parser.error("Commentary anchor falls outside the prepared clip")
            beats = [NarrationBeat(float(line["anchor"]),
                                  (float(lines[i + 1]["anchor"]) if i + 1 < len(lines) else clip_duration) - float(line["anchor"]),
                                  len(line["text"].split()), "", line["text"])
                     for i, line in enumerate(lines)]
        else:
            beats = generate_narration_beats(cues, source_title, clip_duration, model=args.ollama_model,
                                             scene_before=scene_before, scene_after=scene_after)
        narration_txt.write_text(
            "\n".join(f"[{b.anchor:05.1f}s] {b.text}" for b in beats),
            encoding="utf-8",
        )
        print(f"      -> {len(beats)} timed beats -> {narration_txt}")

        step("Placing brief text bylines (no voiceover)..." if args.byline_only else f"Synthesizing narration voiceover (edge-tts, {args.tts_voice})...")
        try:
            if args.byline_only:
                narration_segments = [NarrationSegment(b.text, b.anchor, min(b.anchor + 4.0, clip_duration)) for b in beats]
            else:
                narration_segments = synthesize_narration(beats, narration_wav, clip_duration,
                                                          voice=args.tts_voice, strict=bool(editorial))
        except ValueError as exc:
            # Strict (approved) timing refuses to move or drop a reviewed line —
            # correct, but report it as an error, not a traceback.
            parser.error(str(exc))
        print(f"      -> {len(narration_segments)} lines placed" + (" (text only)" if args.byline_only else f" -> {narration_wav}"))

    step("Using approved title and description..." if editorial else f"Fetching movie context + generating metadata via Ollama ({args.ollama_model})...")
    meta = {k: editorial[k] for k in ("title", "description", "tags")} if editorial else generate_metadata(transcript, source_title=source_title, model=args.ollama_model,
                             content_format="long-form" if layout_mode == "landscape" else "short",
                             editorial_context=" ".join(s.text for s in narration_segments))
    # A title chosen in the web UI wins over the one the LLM just wrote — the
    # description and tags from this call are still used as-is.
    if args.title:
        meta["title"] = args.title
        print(f"      -> title override: {args.title!r}")

    # Description layout: the "edit" style (Movie/Music/Editing Program credits +
    # hashtags + fair-use disclaimer) is the default for recap, opt-in elsewhere.
    # The tool is never named; the editing program is credited as After Effects.
    edit_credits = (args.edit_credits or auto_ranges) and not args.no_edit_credits
    if edit_credits:
        music_credit = args.music_credit
        if not music_credit and args.bg_music:
            music_credit = clean_music_name(Path(args.bg_music).stem)
        styled = build_edit_description(
            meta,
            movie_label=movie_label,
            movie_tag=source_title,
            music=music_credit or "",
            editing_program=args.editing_program,
            disclaimer=not args.no_disclaimer,
        )
        # Put the styled block in both files so whatever consumes metadata.json
        # (e.g. the Agent Hub upload card) shows the same description.
        meta["description"] = styled
        description_path.write_text(styled, encoding="utf-8")
    else:
        description_path.write_text(
            f"{meta['title']}\n\n{meta['description']}",
            encoding="utf-8",
        )
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"      -> {metadata_path}")

    step("Building captions + title card...")
    width, height = frame_size
    if not dialogue_captions and words:
        print(f"      -> dialogue captions off ({len(words)} words kept for the metadata only)")
    build_ass(
        words if dialogue_captions else [], ass_path,
        shape=layout_mode,
        video_width=width, video_height=height,
        title=meta["title"],
        narration=narration_segments,
        layout=layout,
        caption_font=args.caption_font,
        caption_italic=args.caption_italic,
        caption_animation=args.caption_animation,
        caption_shuffle=not args.no_caption_shuffle,
        caption_color=args.caption_color,
        caption_font_shuffle=args.caption_font_shuffle,
    )
    print(f"      -> {ass_path}")

    step("Framing + burning captions into video + mixing audio...")
    # A storytelling narrator is the LEAD track, so the 0.04 default — which
    # exists for commentary, where the voice sits under a film playing at full
    # volume — would bury it ~28 dB down. Bump it unless the user set a level.
    if audio_mix != "original" and abs(args.narration_volume - 0.04) < 1e-9:
        args.narration_volume = 1.0

    # Recap wants a loud music bed — bump the default if the user left it at the
    # classic 0.10 (the web UI sends its own recap value and overrides this).
    # Only meaningful when a track was actually picked.
    if music_leads and recap_music is not None and abs(args.bg_volume - 0.10) < 1e-9:
        args.bg_volume = 0.6
    bg_music_path: Path | None = None
    if not args.no_bg_music:
        if args.bg_music:
            bg_music_path = Path(args.bg_music)
            if not bg_music_path.is_file():
                print(f"      ! --bg-music not found: {bg_music_path}, skipping")
                bg_music_path = None
        elif args.narrate:
            # Movie audio + voiceover + music is muddy — music stays opt-in here
            print("      -> narrated mode: background music off (pass --bg-music to add it)")
        else:
            bg_music_path = pick_random_audio(AUDIO_DIR)
            if bg_music_path:
                print(f"      -> background music: {bg_music_path.name} @ vol {args.bg_volume}")
            else:
                print(f"      -> no audio files in {AUDIO_DIR}, skipping background music")

    watermark_text: str | None = None
    if args.watermark != "none":
        watermark_text = watermarks[args.watermark]
        print(f"      -> watermark: '{watermark_text}' (fades at 3 s)")

    burn_subtitles(
        framing_src, ass_path, final_path,
        bg_music=bg_music_path, bg_volume=args.bg_volume, bg_skip=args.bg_skip,
        watermark=watermark_text,
        narration_audio=narration_wav if narration_segments and not args.byline_only else None,
        narration_volume=args.narration_volume,
        pre_filter=pre_filter,
        mirror=args.mirror,
        music_duck=music_leads and recap_music is not None,
        narration_lead=audio_mix != "original",
        dialogue_volume=0.0 if audio_mix == "narration-only" else args.dialogue_volume,
        output_duration=editorial.get("expected_duration") if editorial and editorial.get("comparison_style") else None,
        loudness_target=(None if args.no_loudnorm else -14.0),  # -14 LUFS master (hotclip/clippyme)
    )
    print(f"      -> {final_path}")
    meta["review"] = {"footage_rights": "unreviewed", "audiovisual": "unreviewed", "studio_checks": "unreviewed"}
    from uuid import uuid4
    meta["render"] = {"render_id": str(uuid4()), "output_path": str(final_path.resolve()),
                      "selection": selection, "voice": voice, "shape": layout_mode,
                      "audio_mix": audio_mix, "source_ranges": ranges,
                      "narration": [{"text": s.text, "start": s.start, "end": s.end} for s in narration_segments]}
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Recap: build the ANIMATED iconic outro (enhanced face stills, beat-timed
    # zoom-out + fade, watermark, music) and append it after the captioned body.
    if spine and recap_shot_times:
        step("Building animated iconic outro (enhanced face stills)...")
        # NON-FATAL: the captioned body is already rendered to final_path. If the
        # outro build or concat fails for any reason, keep the body rather than
        # letting the whole render die — that cascade lost a good body video.
        import subprocess as _sp
        import tempfile as _tf
        import os as _os
        try:
            outro_path = clip_dir / "outro.mp4"
            # Build the outro at the BODY's framerate so the concat below doesn't
            # choke on mismatched fps (the usual reason it "never attached").
            try:
                _fp = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v",
                               "-show_entries", "stream=r_frame_rate", "-of",
                               "default=nw=1:nk=1", str(final_path)],
                              capture_output=True, text=True, check=True)
                _num, _den = _fp.stdout.strip().split("/")
                body_fps = max(1, int(round(float(_num) / float(_den))))
            except Exception:
                body_fps = 30
            build_animated_outro(
                video_path, recap_shot_times, recap_outro_durations, outro_path,
                size=frame_size, fps=body_fps,
                watermark_text=(watermark_text or ""), movie_name=movie_label,
                music_path=bg_music_path, music_start=recap_body_duration,
                music_volume=args.bg_volume, shot_faces=recap_shot_faces,
            )
            # Use the concat FILTER, not the demuxer: the demuxer silently DROPS
            # frames when the two files' framerates differ even slightly (body
            # 24000/1001 vs outro 24/1), leaving a frozen frame + music at the end.
            # The filter re-times both video and audio to a common fps for a clean
            # join with no dropped outro frames and no audio spike.
            merged = clip_dir / "merged.mp4"
            _W, _H = frame_size
            _sp.run(["ffmpeg", "-y", "-i", str(final_path), "-i", str(outro_path),
                     "-filter_complex",
                     f"[0:v]fps={body_fps},scale={_W}:{_H},setsar=1[v0];"
                     f"[1:v]fps={body_fps},scale={_W}:{_H},setsar=1[v1];"
                     f"[v0][v1]concat=n=2:v=1:a=0[v];"
                     f"[0:a][1:a]concat=n=2:v=0:a=1[a]",
                     "-map", "[v]", "-map", "[a]",
                     "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                     "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", str(merged)],
                    check=True, capture_output=True)
            _os.replace(str(merged), str(final_path))
            print(f"      -> appended animated outro -> {final_path}")
        except Exception as exc:
            detail = exc.stderr.decode(errors="replace")[-400:] if isinstance(exc, _sp.CalledProcessError) and exc.stderr else str(exc)
            print(f"      ! animated outro failed, keeping the body video without it:\n        {detail}")

    if want_thumbnail:
        if thumbnail_path.is_file() and args.clip_dir:
            # A thumbnail was hand-picked in the web UI's phase-1 step — keep it.
            step("Using hand-picked thumbnail...")
            print(f"      -> {thumbnail_path}")
        else:
            step(f"Composing {thumb_size[0]}x{thumb_size[1]} thumbnail "
                 "(full frame, title + movie name + border)...")
            # Grab from the uncompressed raw (uncropped) clip so the whole shot shows.
            make_thumbnail(raw_clip_path, thumbnail_path, title=meta["title"], movie=movie_label,
                           width=thumb_size[0], height=thumb_size[1])
            print(f"      -> {thumbnail_path}")

    if args.prepend_thumbnail and thumbnail_path.is_file():
        step("Baking thumbnail into the last 0.1s of the video...")
        width, height = get_resolution(final_path)
        append_still_image(final_path, thumbnail_path, final_path, duration=0.1, width=width, height=height)
        print(f"      -> {final_path} (ends on the thumbnail frame)")

    end_card_added = False
    end_card_error = None
    if args.end_card:
        print("      Building channel end card...", flush=True)
        card_path = clip_dir / "end_card.mp4"
        try:
            build_brand_end_card(card_path, size=frame_size, duration=args.end_card_duration, brand=brand)
            append_end_card(final_path, card_path, duration=args.end_card_duration)
            end_card_added = True
        except Exception as exc:
            end_card_error = exc
            print("      ! End card failed; the rendered body has been preserved", flush=True)

    # Measure the delivered file, after all optional appended segments. Text-only
    # bylines are not spoken narration. Retain earlier metadata if append fails.
    final_duration = get_duration(final_path)
    meta["render"].update({"title": meta["title"],
                          "final_duration": final_duration,
                          "narration_coverage_pct": narration_coverage(narration_segments, final_duration, byline_only=args.byline_only),
                          "byline_only": args.byline_only, "end_card": end_card_added,
                          "end_card_requested": args.end_card,
                          "end_card_duration": args.end_card_duration if end_card_added else 0.0,
                          "youtube_video_id": None})
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if end_card_error is not None:
        detail = getattr(end_card_error, "stderr", None) or str(end_card_error)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        parser.error(f"End card failed; body and metadata retained in {clip_dir}. "
                     f"Check branding/ffmpeg settings or use --no-end-card. {str(detail)[-400:]}")

    print("\nDone.")
    print(f"  Final video:  {final_path}")
    print(f"  Title:        {meta['title']}")
    print(f"  Movie:        {movie_label}")
    print(f"  Metadata:     {metadata_path}")
    print(f"  Description:  {description_path}")
    print(f"  Raw clip:     {raw_clip_path}")
    if args.narrate:
        print(f"  Narration:    {narration_txt}")
    if want_thumbnail and thumbnail_path.is_file():
        print(f"  Thumbnail:    {thumbnail_path}")


if __name__ == "__main__":
    main()
