"""CLI: turn a movie file + chosen timestamp ranges into a captioned YouTube Short with metadata.

Two modes:
- classic: square crop, Whisper captions, bg music (the original pipeline).
- narrated (--narrate): 9:16 vertical layout, SRT-driven captions, an LLM-written
  scene narration read by TTS at low volume with the text shown beneath the video,
  and a thumbnail — the transformative-content format for monetization.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from burn import WATERMARKS, burn_subtitles, pick_random_audio
from captions import build_ass
from clip_selector import cut_and_concat, parse_ranges
from ffutil import get_duration, get_resolution
from metadata import DEFAULT_MODEL, generate_metadata
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
    parser.add_argument("--layout", choices=["square", "vertical"], default=None,
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
    args = parser.parse_args()

    video_arg = args.video.strip().strip('"').strip("'") if args.video else None
    video_path = Path(video_arg) if video_arg else prompt_video_path()
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    ranges = parse_ranges(args.ranges) if args.ranges else prompt_ranges()
    source_title = args.source_title or filename_to_title(video_path)
    layout_mode = args.layout or ("vertical" if args.narrate else "square")

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
    want_thumbnail = args.narrate or args.prepend_thumbnail
    total_steps = (5
                   + (1 if args.remove_silence else 0)
                   + (2 if args.narrate else 0)            # narration beats + voiceover
                   + (1 if want_thumbnail else 0)          # compose/use thumbnail
                   + (1 if args.prepend_thumbnail else 0)) # bake thumbnail into video
    step = StepPrinter(total_steps)

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
        step(f"Cutting {len(ranges)} range(s) from source video...")
        cut_and_concat(video_path, ranges, raw_clip_path)
        print(f"      -> {raw_clip_path}")
    clip_duration = get_duration(raw_clip_path)

    # Optional silence cut: remove spans where the audio is near-zero (no
    # dialogue). Subtitle cue times are remapped through silence_keeps below;
    # the Whisper path needs no remap because it transcribes the trimmed clip.
    framing_src = raw_clip_path
    silence_keeps: list[tuple[float, float]] | None = None
    if args.remove_silence:
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
    else:
        pre_filter = (f"crop=min(iw\\,ih):min(iw\\,ih),"
                      f"scale={args.size}:{args.size}:flags=lanczos")
        frame_size = (args.size, args.size)

    if srt_path:
        step("Extracting dialogue captions from subtitle file...")
        cues = cues_for_ranges(parse_srt(srt_path), ranges)
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
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    description_path.write_text(
        f"{meta['title']}\n\n{meta['description']}",
        encoding="utf-8",
    )
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
    )
    print(f"      -> {final_path}")

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
