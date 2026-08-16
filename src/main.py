"""CLI: turn a movie file + chosen timestamp ranges into a captioned YouTube Short with metadata.

Three modes (--mode, or the legacy --narrate):
- classic: square crop, Whisper captions, bg music (the original pipeline).
- narrated: 9:16 vertical layout, SRT-driven captions, an LLM-written scene
  narration read by TTS at low volume with the text shown beneath the video,
  and a thumbnail — the transformative-content format for monetization.
- recap: a plot-recap montage. The body ranges (picked by recap.py) are cut to
  the hand-picked music's beats, the movie dialogue is ducked under a loud music
  bed, and the final few seconds are a beat-synced montage of iconic face shots
  found by scanning the film.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from burn import WATERMARKS, burn_subtitles, pick_random_audio
from captions import build_ass
from clip_selector import cut_and_concat, parse_ranges, parse_timestamp
from ffutil import get_duration, get_resolution
from outro import (
    _outro_durations,
    build_animated_outro,
    detect_beats,
    find_iconic_shots,
    snap_cuts_to_beats,
)
from recap import suggest_recap
from metadata import (
    DEFAULT_EDITING_PROGRAM,
    DEFAULT_MODEL,
    build_edit_description,
    clean_music_name,
    generate_metadata,
)
from narration import generate_narration_beats
from reformat import append_still_image
from silence import detect_silences, keep_intervals, remap_time
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
from tts import DEFAULT_VOICE, synthesize_narration

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
    parser.add_argument("--video", help="Path to the source movie file (skips the prompt)")
    parser.add_argument("--ranges", help="Timestamp ranges, e.g. '3:20-3:30; 5:15-5:45' (skips the prompt)")
    parser.add_argument("--whisper-model", default="medium", help="faster-whisper model size (default: medium)")
    parser.add_argument("--language", default="en", help="Transcription language code (default: en)")
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL, help=f"Ollama model for metadata (default: {DEFAULT_MODEL})")
    parser.add_argument("--source-title", help="Movie/show name override (default: parsed from the filename)")
    parser.add_argument("--title", default=None,
                        help="On-screen title override (default: written by the LLM in the metadata step)")
    parser.add_argument("--size", type=int, default=1080, help="Square-mode side length in pixels (default: 1080)")
    parser.add_argument("--bg-music", default=None, help="Path to background music file (default: random from audio/)")
    parser.add_argument("--bg-volume", type=float, default=0.10, help="Background music volume 0.0–1.0 (default: 0.10)")
    parser.add_argument("--bg-skip", type=float, default=0.0,
                        help="Skip the first N seconds of the background music (e.g. cut a slow intro). Default: 0")
    parser.add_argument("--no-bg-music", action="store_true", help="Disable background music entirely")
    parser.add_argument(
        "--watermark",
        default="mv-edits",
        choices=list(WATERMARKS.keys()) + ["none"],
        help="Watermark to burn in (bottom centre, fades after 3 s). Use 'none' to disable. (default: mv-edits)",
    )
    # ── narrated-format options ──────────────────────────────────────────────
    parser.add_argument("--narrate", action="store_true",
                        help="Narrated vertical format: LLM scene narration read by TTS + text beneath the video")
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
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror (flip left-right) the clip footage only — captions/narration text stay upright")
    # ── recap-format options ─────────────────────────────────────────────────
    parser.add_argument("--mode", choices=["classic", "narrated", "recap"], default=None,
                        help="Output format. 'recap' = beat-synced plot-recap montage. Overrides --narrate.")
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

    video_arg = args.video.strip().strip('"').strip("'") if args.video else None
    video_path = Path(video_arg) if video_arg else prompt_video_path()
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    ranges = parse_ranges(args.ranges) if args.ranges else []
    source_title = args.source_title or filename_to_title(video_path)

    # Format resolution: --mode wins; --narrate is the legacy alias for narrated.
    mode = args.mode or ("narrated" if args.narrate else "classic")
    is_recap = mode == "recap"
    if is_recap:
        args.narrate = False  # recap has no TTS narration; it's music + ducked dialogue
    layout_mode = args.layout or ("vertical" if (args.narrate or is_recap) else "square")

    # Non-recap needs explicit ranges (prompt when interactive). Recap with no
    # ranges auto-builds them from the whole subtitle file further down — "if
    # nothing is given, default to the full movie".
    if not ranges and not is_recap:
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

    # Recap mode has two hard requirements: the subtitle file (it drives both the
    # beat picks and the burned captions) and a music track (the whole edit is
    # cut to its beats). Fail fast with a clear message rather than half-render.
    recap_music: Path | None = None
    if is_recap:
        if srt_path is None:
            print("Error: recap mode needs a subtitle file (--srt) — it drives the captions.")
            sys.exit(1)
        if not args.bg_music:
            print("Error: recap mode needs a music track (--bg-music) — the edit is cut to its beats.")
            sys.exit(1)
        recap_music = Path(args.bg_music)
        if not recap_music.is_file():
            print(f"Error: music track not found: {recap_music}")
            sys.exit(1)

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
        cut_and_concat(video_path, ranges, raw_clip_path)
        (clip_dir / "prepare.json").write_text(
            json.dumps(_prepare_fingerprint(video_path, ranges), indent=2), encoding="utf-8")
        print("\nPrepared.")
        print(f"  Clip dir:     {clip_dir}")
        print(f"  Raw clip:     {raw_clip_path}")
        print(f"  Movie:        {movie_label}")
        return

    # 5 base steps: cut, captions, metadata, build-ass, burn. Framing (square /
    # vertical canvas) is folded into the burn encode, not a step of its own.
    # A thumbnail is composed whenever the narrated format needs one OR it's
    # being baked into the video (--prepend-thumbnail), so every pick-first run —
    # classic included — accounts for the compose + bake steps.
    want_thumbnail = args.narrate or args.prepend_thumbnail or is_recap
    total_steps = (5
                   + (2 if is_recap else 0)                # recap: beat/shot analysis + animated outro
                   + (1 if (args.remove_silence and not is_recap) else 0)
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
    if is_recap:
        step("Building recap from subtitles + analyzing music beats + iconic shots...")
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
        regions = [(s, e) for s, e in ranges if e - s > 30.0]
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
        beats = detect_beats(recap_music)
        snapped_body = snap_cuts_to_beats(ranges, beats)
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
        print(f"      -> {len(beats)} beats | {len(snapped_body)} body clips snapped "
              f"(~{recap_body_duration:.0f}s) | {len(recap_shot_times)} outro slow-mo clip(s)")

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

    if reuse_ok:
        step(f"Reusing prepared raw clip ({raw_clip_path.name})...")
        print(f"      -> {raw_clip_path}")
    else:
        step(f"Cutting {len(cut_ranges)} range(s) from source video...")
        cut_and_concat(video_path, cut_ranges, raw_clip_path, fade_edges=is_recap)
        print(f"      -> {raw_clip_path}")
    clip_duration = get_duration(raw_clip_path)

    # Optional silence cut: remove spans where the audio is near-zero (no
    # dialogue). Subtitle cue times are remapped through silence_keeps below;
    # the Whisper path needs no remap because it transcribes the trimmed clip.
    framing_src = raw_clip_path
    silence_keeps: list[tuple[float, float]] | None = None
    if args.remove_silence and not is_recap:
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

    narration_segments: list = []
    if args.narrate:
        step(f"Writing timed scene narration via Ollama ({args.ollama_model})...")
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
        beats = generate_narration_beats(cues, source_title, clip_duration, model=args.ollama_model,
                                         scene_before=scene_before, scene_after=scene_after)
        narration_txt.write_text(
            "\n".join(f"[{b.anchor:05.1f}s] {b.text}" for b in beats),
            encoding="utf-8",
        )
        print(f"      -> {len(beats)} timed beats -> {narration_txt}")

        step(f"Synthesizing narration voiceover (edge-tts, {args.tts_voice})...")
        narration_segments = synthesize_narration(beats, narration_wav, clip_duration, voice=args.tts_voice)
        print(f"      -> {len(narration_segments)} lines placed -> {narration_wav}")

    step(f"Fetching movie context + generating metadata via Ollama ({args.ollama_model})...")
    meta = generate_metadata(transcript, source_title=source_title, model=args.ollama_model)
    # A title chosen in the web UI wins over the one the LLM just wrote — the
    # description and tags from this call are still used as-is.
    if args.title:
        meta["title"] = args.title
        print(f"      -> title override: {args.title!r}")

    # Description layout: the "edit" style (Movie/Music/Editing Program credits +
    # hashtags + fair-use disclaimer) is the default for recap, opt-in elsewhere.
    # The tool is never named; the editing program is credited as After Effects.
    edit_credits = (args.edit_credits or is_recap) and not args.no_edit_credits
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
    build_ass(
        words, ass_path,
        video_width=width, video_height=height,
        title=meta["title"],
        narration=narration_segments,
        layout=layout,
        caption_font=args.caption_font,
        caption_animation=args.caption_animation,
        caption_shuffle=not args.no_caption_shuffle,
        caption_color=args.caption_color,
    )
    print(f"      -> {ass_path}")

    step("Framing + burning captions into video + mixing audio...")
    # Recap wants a loud music bed — bump the default if the user left it at the
    # classic 0.10 (the web UI sends its own recap value and overrides this).
    if is_recap and abs(args.bg_volume - 0.10) < 1e-9:
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
        watermark_text = WATERMARKS[args.watermark]
        print(f"      -> watermark: '{watermark_text}' (fades at 3 s)")

    burn_subtitles(
        framing_src, ass_path, final_path,
        bg_music=bg_music_path, bg_volume=args.bg_volume, bg_skip=args.bg_skip,
        watermark=watermark_text,
        narration_audio=narration_wav if narration_segments else None,
        narration_volume=args.narration_volume,
        pre_filter=pre_filter,
        mirror=args.mirror,
        music_duck=is_recap,
    )
    print(f"      -> {final_path}")

    # Recap: build the ANIMATED iconic outro (enhanced face stills, beat-timed
    # zoom-out + fade, watermark, music) and append it after the captioned body.
    if is_recap and recap_shot_times:
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
                watermark_text=(watermark_text or "MV EDITS"), movie_name=movie_label,
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
            step("Composing 9:16 thumbnail (full frame, title + movie name + border)...")
            # Grab from the uncompressed raw (uncropped) clip so the whole shot shows.
            make_thumbnail(raw_clip_path, thumbnail_path, title=meta["title"], movie=movie_label)
            print(f"      -> {thumbnail_path}")

    if args.prepend_thumbnail and thumbnail_path.is_file():
        step("Baking thumbnail into the last 0.1s of the video...")
        width, height = get_resolution(final_path)
        append_still_image(final_path, thumbnail_path, final_path, duration=0.1, width=width, height=height)
        print(f"      -> {final_path} (ends on the thumbnail frame)")

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
