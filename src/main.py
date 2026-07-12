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
from reformat import append_still_image, compose_vertical, crop_to_square
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
    r"1080p|720p|2160p|4k|uhd|hdr|webrip|web-dl|webdl|bluray|blu-ray|brrip|dvdrip|"
    r"hdtv|x264|x265|h264|h265|hevc|aac\d?|ac3|dts|yts|yify|rarbg|amzn|nf|repack|"
    r"proper|extended|remastered|directors cut|multi|dual audio"
    r")\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "clip"


def filename_to_title(path: Path) -> str:
    """Best-effort cleanup of a movie/TV filename into a human-readable title."""
    stem = path.stem
    stem = re.sub(r"-[A-Z0-9]{2,}$", "", stem)  # trailing release-group tag, e.g. "-RARBG"
    stem = re.sub(r"[\[\(\{].*?[\]\)\}]", " ", stem)
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


class StepPrinter:
    """Prints '[n/N] ...' progress lines (the web UI parses these)."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.n = 0

    def __call__(self, message: str) -> None:
        self.n += 1
        print(f"\n[{self.n}/{self.total}] {message}" if self.n == 1 else f"[{self.n}/{self.total}] {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a captioned movie short with AI metadata.")
    parser.add_argument("--video", help="Path to the source movie file (skips the prompt)")
    parser.add_argument("--ranges", help="Timestamp ranges, e.g. '3:20-3:30; 5:15-5:45' (skips the prompt)")
    parser.add_argument("--whisper-model", default="medium", help="faster-whisper model size (default: medium)")
    parser.add_argument("--language", default="en", help="Transcription language code (default: en)")
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL, help=f"Ollama model for metadata (default: {DEFAULT_MODEL})")
    parser.add_argument("--source-title", help="Movie/show name override (default: parsed from the filename)")
    parser.add_argument("--size", type=int, default=1080, help="Square-mode side length in pixels (default: 1080)")
    parser.add_argument("--bg-music", default=None, help="Path to background music file (default: random from audio/)")
    parser.add_argument("--bg-volume", type=float, default=0.10, help="Background music volume 0.0–1.0 (default: 0.10)")
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
    parser.add_argument("--narration-volume", type=float, default=0.05,
                        help="Narration voiceover volume 0.0–1.0 (default: 0.05)")
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
    framed_path     = clip_dir / ("clip_vertical.mp4" if layout_mode == "vertical" else "clip_square.mp4")
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
        print("\nPrepared.")
        print(f"  Clip dir:     {clip_dir}")
        print(f"  Raw clip:     {raw_clip_path}")
        print(f"  Movie:        {movie_label}")
        return

    total_steps = 6 + (3 if args.narrate else 0) + (1 if args.prepend_thumbnail else 0)
    step = StepPrinter(total_steps)

    if raw_clip_path.is_file() and args.clip_dir:
        step(f"Reusing prepared raw clip ({raw_clip_path.name})...")
        print(f"      -> {raw_clip_path}")
    else:
        step(f"Cutting {len(ranges)} range(s) from source video...")
        cut_and_concat(video_path, ranges, raw_clip_path)
        print(f"      -> {raw_clip_path}")
    clip_duration = get_duration(raw_clip_path)

    layout: dict | None = None
    if layout_mode == "vertical":
        step("Composing 9:16 vertical layout (square crop on black canvas)...")
        _, layout = compose_vertical(raw_clip_path, framed_path)
        print(f"      -> {framed_path} (video band {layout['video_top']}–{layout['video_bottom']}px)")
    else:
        step(f"Cropping to {args.size}x{args.size} square (center crop)...")
        crop_to_square(raw_clip_path, framed_path, size=args.size)
        print(f"      -> {framed_path}")

    if srt_path:
        step("Extracting dialogue captions from subtitle file...")
        cues = cues_for_ranges(parse_srt(srt_path), ranges)
        words = cues_to_words(cues)
        transcript = cues_to_text(cues)
        print(f"      -> {len(cues)} cues / {len(words)} words from {srt_path.name}")
    else:
        step("Transcribing clip with faster-whisper (no subtitle file found)...")
        words = transcribe(framed_path, model_size=args.whisper_model, language=args.language, source_title=source_title)
        transcript = words_to_text(words)
        cues = cues_from_words(words)
        print(f"      -> {len(words)} words transcribed")

    narration_segments: list = []
    if args.narrate:
        step(f"Writing timed scene narration via Ollama ({args.ollama_model})...")
        beats = generate_narration_beats(cues, source_title, clip_duration, model=args.ollama_model)
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
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    description_path.write_text(
        f"{meta['title']}\n\n{meta['description']}",
        encoding="utf-8",
    )
    print(f"      -> {metadata_path}")

    step("Building captions + title card...")
    width, height = get_resolution(framed_path)
    build_ass(
        words, ass_path,
        video_width=width, video_height=height,
        title=meta["title"],
        narration=narration_segments,
        layout=layout,
    )
    print(f"      -> {ass_path}")

    step("Burning captions into video + mixing audio...")
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
        framed_path, ass_path, final_path,
        bg_music=bg_music_path, bg_volume=args.bg_volume,
        watermark=watermark_text,
        narration_audio=narration_wav if narration_segments else None,
        narration_volume=args.narration_volume,
    )
    print(f"      -> {final_path}")

    if args.narrate:
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
        print(f"  Thumbnail:    {thumbnail_path}")


if __name__ == "__main__":
    main()
