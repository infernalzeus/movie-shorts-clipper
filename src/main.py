"""CLI: turn a movie file + chosen timestamp ranges into a captioned YouTube Short with metadata."""

import argparse
import json
import re
import sys
from pathlib import Path

from burn import WATERMARKS, burn_subtitles, pick_random_audio
from captions import build_ass
from clip_selector import cut_and_concat, parse_ranges
from ffutil import get_resolution
from metadata import DEFAULT_MODEL, generate_metadata
from reformat import crop_to_square
from transcriber import transcribe, words_to_text

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a captioned movie short with AI metadata.")
    parser.add_argument("--video", help="Path to the source movie file (skips the prompt)")
    parser.add_argument("--ranges", help="Timestamp ranges, e.g. '3:20-3:30; 5:15-5:45' (skips the prompt)")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size (default: small)")
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL, help=f"Ollama model for metadata (default: {DEFAULT_MODEL})")
    parser.add_argument("--source-title", help="Movie/show name override (default: parsed from the filename)")
    parser.add_argument("--size", type=int, default=1080, help="Output square side length in pixels (default: 1080)")
    parser.add_argument("--bg-music", default=None, help="Path to background music file (default: random from audio/)")
    parser.add_argument("--bg-volume", type=float, default=0.10, help="Background music volume 0.0–1.0 (default: 0.10)")
    parser.add_argument("--no-bg-music", action="store_true", help="Disable background music entirely")
    parser.add_argument(
        "--watermark",
        default="mv-edits",
        choices=list(WATERMARKS.keys()) + ["none"],
        help="Watermark to burn in (bottom centre, fades after 3 s). Use 'none' to disable. (default: mv-edits)",
    )
    args = parser.parse_args()

    video_path = Path(args.video) if args.video else prompt_video_path()
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    ranges = parse_ranges(args.ranges) if args.ranges else prompt_ranges()
    source_title = args.source_title or filename_to_title(video_path)

    slug = slugify(video_path.stem)[:10]
    clip_dir = OUTPUT_DIR / slug
    if clip_dir.exists():
        n = 2
        while (OUTPUT_DIR / f"{slug}-{n}").exists():
            n += 1
        clip_dir = OUTPUT_DIR / f"{slug}-{n}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    raw_clip_path = clip_dir / "clip_raw.mp4"
    square_clip_path = clip_dir / "clip_square.mp4"
    ass_path = clip_dir / "captions.ass"
    final_path = clip_dir / "clip_final.mp4"
    metadata_path = clip_dir / "metadata.json"
    description_path = clip_dir / "description.txt"

    print(f"Source title detected: {source_title!r}")

    print(f"\n[1/6] Cutting {len(ranges)} range(s) from source video...")
    cut_and_concat(video_path, ranges, raw_clip_path)
    print(f"      -> {raw_clip_path}")

    print(f"[2/6] Cropping to {args.size}x{args.size} square (center crop)...")
    crop_to_square(raw_clip_path, square_clip_path, size=args.size)
    print(f"      -> {square_clip_path}")

    print("[3/6] Transcribing clip with faster-whisper...")
    words = transcribe(square_clip_path, model_size=args.whisper_model)
    transcript = words_to_text(words)
    print(f"      -> {len(words)} words transcribed")

    print(f"[4/6] Fetching movie context + generating metadata via Ollama ({args.ollama_model})...")
    meta = generate_metadata(transcript, source_title=source_title, model=args.ollama_model)
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    description_path.write_text(
        f"{meta['title']}\n\n{meta['description']}",
        encoding="utf-8",
    )
    print(f"      -> {metadata_path}")

    print("[5/6] Building sentence captions + title card...")
    width, height = get_resolution(square_clip_path)
    build_ass(words, ass_path, video_width=width, video_height=height, title=meta["title"])
    print(f"      -> {ass_path}")

    print("[6/6] Burning captions into video...")
    bg_music_path: Path | None = None
    if not args.no_bg_music:
        if args.bg_music:
            bg_music_path = Path(args.bg_music)
            if not bg_music_path.is_file():
                print(f"      ! --bg-music not found: {bg_music_path}, skipping")
                bg_music_path = None
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
        square_clip_path, ass_path, final_path,
        bg_music=bg_music_path, bg_volume=args.bg_volume,
        watermark=watermark_text,
    )
    print(f"      -> {final_path}")

    print("\nDone.")
    print(f"  Final video:  {final_path}")
    print(f"  Title:        {meta['title']}")
    print(f"  Metadata:     {metadata_path}")
    print(f"  Description:  {description_path}")


if __name__ == "__main__":
    main()
