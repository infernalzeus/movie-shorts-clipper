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
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def _escape_filter_path(path: Path) -> str:
    p = path.resolve().as_posix()
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


# Selectable poster fonts (Windows built-ins verified on this machine). The
# key is what the web UI / CLI passes; applied to BOTH title and movie text.
FONTS: dict[str, str] = {
    "impact":       "C:/Windows/Fonts/impact.ttf",
    "cooper-black": "C:/Windows/Fonts/COOPBL.TTF",     # chunky retro
    "showcard":     "C:/Windows/Fonts/SHOWG.TTF",      # showbiz poster
    "bernard":      "C:/Windows/Fonts/BERNHC.TTF",     # tall condensed poster
    "bauhaus":      "C:/Windows/Fonts/BAUHS93.TTF",    # funky 70s
    "broadway":     "C:/Windows/Fonts/broadw.ttf",     # art deco
    "stencil":      "C:/Windows/Fonts/STENCIL.TTF",    # military
    "rockwell":     "C:/Windows/Fonts/ROCKB.TTF",      # slab serif
    "bahnschrift":  "C:/Windows/Fonts/bahnschrift.ttf",# clean DIN
    "segoe-black":  "C:/Windows/Fonts/seguibl.ttf",
    "arial-black":  "C:/Windows/Fonts/ariblk.ttf",
    "georgia":      "C:/Windows/Fonts/georgiab.ttf",   # elegant serif
}
DEFAULT_FONT = "impact"

# Named text colors (any raw ffmpeg color string is also accepted).
COLORS: dict[str, str] = {
    "white":  "white",
    "gold":   "#FFD400",
    "red":    "#FF4136",
    "orange": "#FF7A30",
    "cyan":   "#00E5FF",
    "lime":   "#3DDC84",
    "pink":   "#FF2D95",
}

_FALLBACK_FONTS = [
    FONTS["impact"], FONTS["arial-black"], "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _resolve_font(font: str | None) -> Path | None:
    candidates = ([FONTS[font]] if font and font in FONTS else []) + _FALLBACK_FONTS
    for candidate in candidates:
        p = Path(candidate)
        if p.is_file():
            return p
    return None


def _find_matching_time(
    source: Path,
    ref_image: Path,
    around: float,
    window: float = 2.0,
    fps: int = 12,
) -> float | None:
    """Find the timestamp near `around` in `source` whose frame best matches
    ref_image (a canvas capture of the frame the user picked in the browser).

    Decodes a small window of low-res grayscale frames and picks the minimum
    mean-absolute-difference. Returns None when nothing matches confidently —
    e.g. the mapped time is wrong because ranges were edited after preparing.
    """
    import numpy as np

    w, h = 96, 54
    start = max(0.0, around - window / 2)
    frames_raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{window:.3f}",
         "-i", str(source), "-vf", f"fps={fps},scale={w}:{h}",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True,
    ).stdout
    ref_raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(ref_image),
         "-vf", f"scale={w}:{h}", "-pix_fmt", "gray",
         "-frames:v", "1", "-f", "rawvideo", "-"],
        capture_output=True,
    ).stdout

    fsize = w * h
    n = len(frames_raw) // fsize
    if n == 0 or len(ref_raw) < fsize:
        return None
    frames = np.frombuffer(frames_raw[:n * fsize], dtype=np.uint8).reshape(n, fsize).astype(np.int16)
    ref = np.frombuffer(ref_raw[:fsize], dtype=np.uint8).astype(np.int16)
    mae = np.abs(frames - ref).mean(axis=1)
    best = int(mae.argmin())
    if float(mae[best]) > 20.0:  # nothing plausibly the same frame
        return None
    return start + (best + 0.5) / fps


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
    image: Path | None = None,
    font: str | None = None,
    color: str = "white",
    match_image: Path | None = None,
) -> Path:
    """Write a full-bleed 9:16 .jpg thumbnail from source_video.

    image a still image to use as the frame instead of grabbing one from the
          video — the web UI sends the browser's exactly-displayed frame this
          way (Safari's scrub preview snaps to keyframes, so a timestamp grab
          can land on a different frame than the one the user was looking at).
    at    seconds to grab the frame from; if None (and no image), ffmpeg's
          `thumbnail` filter picks a representative one.
    pan   horizontal position (0.0 = left edge, 1.0 = right edge) of the 9:16
          crop window within the wider frame.
    title overlaid across the top; movie overlaid across the bottom; a lined
    border frames the whole thing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pan = min(1.0, max(0.0, pan))

    # Prefer grabbing the frame from the ORIGINAL source at full quality: the
    # reference image (a browser canvas capture of the compressed clip) only
    # identifies WHICH frame the user picked; the pixels come from the source.
    if match_image is not None:
        matched = _find_matching_time(source_video, match_image, at) if at is not None else None
        if matched is not None:
            at, image = matched, None
        else:
            # No confident match (ranges changed? capture failed?) — use the
            # captured image itself so the picked frame is still honored.
            at, image = None, match_image

    # Cover the 9:16 canvas (no bars), then crop to the chosen horizontal
    # window. Lanczos for crisp downscale; a light sharpen + saturation pop
    # (before the border/text so those stay clean).
    steps = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={width}:{height}:x=(in_w-out_w)*{pan:.4f}:y=(in_h-out_h)/2",
        "unsharp=5:5:0.5,eq=saturation=1.12:contrast=1.03",
    ]
    if at is None and image is None:
        steps.insert(0, "thumbnail=250")

    # No darkening scrims — the overlaid title/movie rely on a heavy black outline
    # and drop shadow for legibility, keeping the image itself fully visible.
    # Double lined border, inset well away from the canvas edge.
    steps.append(f"drawbox=x=64:y=64:w={width-128}:h={height-128}:color=white@0.85:t=4")
    steps.append(f"drawbox=x=82:y=82:w={width-164}:h={height-164}:color=white@0.35:t=2")

    text_color = COLORS.get(color, color or "white")
    text_font = _resolve_font(font)
    font_arg = f":fontfile='{_escape_filter_path(text_font)}'" if text_font else ""

    with tempfile.TemporaryDirectory(prefix="thumb-") as tmp:
        tmp_dir = Path(tmp)

        def _fit_size(wrapped: str, base: int) -> int:
            # Shrink the font when the longest line would spill past the border
            # (wide poster fonts like Cooper Black average ~0.74 of the font
            # size per character). Never grow past the base size.
            longest = max(len(line) for line in wrapped.split("\n"))
            return min(base, int(width * 0.90 / (0.74 * max(longest, 1))))

        if title:
            wrapped_title = _wrap(title.upper(), width=20, max_lines=3)
            tf = tmp_dir / "title.txt"
            tf.write_text(wrapped_title, encoding="utf-8")
            # Smaller than the movie name — this is the hook line, not the headline.
            steps.append(
                f"drawtext=textfile='{_escape_filter_path(tf)}'{font_arg}"
                f":fontcolor={text_color}:fontsize={_fit_size(wrapped_title, max(46, width // 20))}"
                f":borderw=5:bordercolor=black:shadowcolor=black@0.6:shadowx=3:shadowy=3"
                f":line_spacing=14:text_align=center"
                f":x=(w-text_w)/2:y=165"
            )

        if movie:
            # Split a trailing "(...)" — the release year or season — onto its own
            # smaller line, anchored a fixed distance above the bottom so it can
            # never fall off the frame below the movie name (the old single
            # multi-line block relied on text_h, which under-counts line spacing
            # and pushed the second line off-screen).
            m = re.match(r"^(.*?)\s*(\([^)]*\))\s*$", movie.strip())
            if m and m.group(1).strip():
                movie_main, movie_sub = m.group(1).strip(), m.group(2).strip()
            else:
                movie_main, movie_sub = movie.strip(), ""

            wrapped_main = _wrap(movie_main.upper(), width=16, max_lines=2)
            mf = tmp_dir / "movie.txt"
            mf.write_text(wrapped_main, encoding="utf-8")
            main_size = _fit_size(wrapped_main, max(84, width // 9))

            bottom_pad = 200  # keep the lowest line this far above the frame edge
            if movie_sub:
                sub_size = max(38, width // 26)
                sf = tmp_dir / "movie_sub.txt"
                sf.write_text(movie_sub.upper(), encoding="utf-8")
                steps.append(
                    f"drawtext=textfile='{_escape_filter_path(sf)}'{font_arg}"
                    f":fontcolor={text_color}:fontsize={sub_size}"
                    f":borderw=5:bordercolor=black:shadowcolor=black@0.7:shadowx=3:shadowy=3"
                    f":text_align=center:x=(w-text_w)/2:y=h-th-{bottom_pad}"
                )
                name_y = f"h-th-{bottom_pad + sub_size + 34}"
            else:
                name_y = f"h-th-{bottom_pad}"

            # Large word-art for the movie name (sits above the year line).
            steps.append(
                f"drawtext=textfile='{_escape_filter_path(mf)}'{font_arg}"
                f":fontcolor={text_color}:fontsize={main_size}"
                f":borderw=7:bordercolor=black:shadowcolor=black@0.7:shadowx=4:shadowy=4"
                f":line_spacing=6:text_align=center"
                f":x=(w-text_w)/2:y={name_y}"
            )

        cmd = ["ffmpeg", "-y"]
        if image is not None:
            cmd += ["-i", str(image)]
        else:
            if at is not None:
                cmd += ["-ss", f"{max(0.0, at):.3f}"]
            cmd += ["-i", str(source_video)]
        cmd += [
            "-vf", ",".join(steps),
            "-frames:v", "1",
            "-q:v", "1",
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
    parser.add_argument("--image", default=None, help="Use this still image as the frame instead of grabbing from the video")
    parser.add_argument("--match-image", default=None,
                        help="Reference image of the picked frame: grab the best-matching frame near --at from --video (full source quality)")
    parser.add_argument("--font", default=DEFAULT_FONT, choices=list(FONTS),
                        help=f"Text font for title + movie name (default: {DEFAULT_FONT})")
    parser.add_argument("--color", default="white",
                        help=f"Text color: {', '.join(COLORS)} or any ffmpeg color (default: white)")
    parser.add_argument("--pan", type=float, default=0.5, help="Horizontal crop position 0.0–1.0 (default 0.5)")
    parser.add_argument("--title", default=None, help="Title text (overlaid top)")
    parser.add_argument("--movie", default=None, help="Movie name (overlaid bottom)")
    args = parser.parse_args()

    src = Path(args.video)
    if not src.is_file():
        print(f"Error: source video not found: {src}", file=sys.stderr)
        sys.exit(1)

    image = Path(args.image) if args.image else None
    if image is not None and not image.is_file():
        print(f"Error: frame image not found: {image}", file=sys.stderr)
        sys.exit(1)
    match_image = Path(args.match_image) if args.match_image else None
    if match_image is not None and not match_image.is_file():
        print(f"Error: match image not found: {match_image}", file=sys.stderr)
        sys.exit(1)

    out = make_thumbnail(src, Path(args.out), title=args.title, movie=args.movie,
                         at=args.at, pan=args.pan, image=image,
                         font=args.font, color=args.color, match_image=match_image)
    print(str(out))


if __name__ == "__main__":
    main()
