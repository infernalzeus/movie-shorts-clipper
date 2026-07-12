"""Compose a full-bleed 9:16 thumbnail from a chosen movie frame.

The frame is cropped to fill the whole 9:16 canvas (no black bars), with the
title overlaid across the top and the movie name across the bottom over subtle
scrims, framed by a lined border. Because a wide frame has to be cropped to a
tall 9:16 window, the horizontal position of that window (`pan`) is chosen in
the web UI so the subject stays in frame.

Runnable as a CLI so the web UI can regenerate the thumbnail from a picked frame:
    python thumbnail.py --video clip_raw.mp4 --out thumbnail.jpg --at 12.3 \
        --pan 0.5 --title "TEA TIME" --movie "Legend"
"""

import argparse
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def _escape_filter_path(path: Path) -> str:
    p = path.resolve().as_posix()
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


# Bold, condensed "poster" fonts for the movie-title word art, in preference order.
_WORD_ART_FONTS = [
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/ariblk.ttf",     # Arial Black
    "C:/Windows/Fonts/arialbd.ttf",    # Arial Bold
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _word_art_font() -> Path | None:
    for candidate in _WORD_ART_FONTS:
        p = Path(candidate)
        if p.is_file():
            return p
    return None


def _wrap(text: str, width: int, max_lines: int) -> str:
    lines = textwrap.wrap(text.strip(), width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "…"
    return "\n".join(lines)


def make_thumbnail(
    source_video: Path,
    output_path: Path,
    title: str | None = None,
    movie: str | None = None,
    at: float | None = None,
    pan: float = 0.5,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """Write a full-bleed 9:16 .jpg thumbnail from source_video.

    at    seconds to grab the frame from; if None, ffmpeg's `thumbnail` filter
          picks a representative one.
    pan   horizontal position (0.0 = left edge, 1.0 = right edge) of the 9:16
          crop window within the wider frame.
    title overlaid across the top; movie overlaid across the bottom; a lined
    border frames the whole thing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pan = min(1.0, max(0.0, pan))

    # Cover the 9:16 canvas (no bars), then crop to the chosen horizontal window.
    steps = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}:x=(in_w-out_w)*{pan:.4f}:y=(in_h-out_h)/2",
    ]
    if at is None:
        steps.insert(0, "thumbnail=250")

    # No darkening scrims — the overlaid title/movie rely on a heavy black outline
    # and drop shadow for legibility, keeping the image itself fully visible.
    # Double lined border, inset from the edge.
    steps.append(f"drawbox=x=18:y=18:w={width-36}:h={height-36}:color=white@0.85:t=4")
    steps.append(f"drawbox=x=30:y=30:w={width-60}:h={height-60}:color=white@0.35:t=2")

    with tempfile.TemporaryDirectory(prefix="thumb-") as tmp:
        tmp_dir = Path(tmp)

        if title:
            tf = tmp_dir / "title.txt"
            tf.write_text(_wrap(title.upper(), width=18, max_lines=3), encoding="utf-8")
            steps.append(
                f"drawtext=textfile='{_escape_filter_path(tf)}'"
                f":fontcolor=white:fontsize={max(58, width // 15)}"
                f":borderw=6:bordercolor=black:shadowcolor=black@0.6:shadowx=3:shadowy=3"
                f":line_spacing=14:text_align=center"
                f":x=(w-text_w)/2:y=120"
            )

        if movie:
            mf = tmp_dir / "movie.txt"
            mf.write_text(_wrap(movie.upper(), width=16, max_lines=2), encoding="utf-8")
            # Large bold word-art for the movie title (with release year).
            art_font = _word_art_font()
            font_arg = f":fontfile='{_escape_filter_path(art_font)}'" if art_font else ""
            steps.append(
                f"drawtext=textfile='{_escape_filter_path(mf)}'{font_arg}"
                f":fontcolor=white:fontsize={max(84, width // 9)}"
                f":borderw=7:bordercolor=black:shadowcolor=black@0.7:shadowx=4:shadowy=4"
                f":line_spacing=6:text_align=center"
                f":x=(w-text_w)/2:y=h-text_h-90"
            )

        cmd = ["ffmpeg", "-y"]
        if at is not None:
            cmd += ["-ss", f"{max(0.0, at):.3f}"]
        cmd += [
            "-i", str(source_video),
            "-vf", ",".join(steps),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(f"ffmpeg thumbnail failed:\n{stderr}") from None

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose a full-bleed 9:16 thumbnail from a movie frame.")
    parser.add_argument("--video", required=True, help="Source clip to grab the frame from")
    parser.add_argument("--out", required=True, help="Output .jpg path")
    parser.add_argument("--at", type=float, default=None, help="Timestamp (seconds) to grab; omit to auto-pick")
    parser.add_argument("--pan", type=float, default=0.5, help="Horizontal crop position 0.0–1.0 (default 0.5)")
    parser.add_argument("--title", default=None, help="Title text (overlaid top)")
    parser.add_argument("--movie", default=None, help="Movie name (overlaid bottom)")
    args = parser.parse_args()

    src = Path(args.video)
    if not src.is_file():
        print(f"Error: source video not found: {src}", file=sys.stderr)
        sys.exit(1)

    out = make_thumbnail(src, Path(args.out), title=args.title, movie=args.movie, at=args.at, pan=args.pan)
    print(str(out))


if __name__ == "__main__":
    main()
