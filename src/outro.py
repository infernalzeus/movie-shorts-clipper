"""Beat-synced iconic outro + whole-recap beat snapping for the Recap Montage format.

Two jobs, both driven by the hand-picked music track:

1. Beat detection (`detect_beats`) — librosa finds the music's beat grid. The
   recap's cuts are nudged onto that grid so every shot change lands on a beat
   (`snap_cuts_to_beats`), and the closing montage is cut one flash per beat
   (`build_outro_flashes`).

2. Iconic-shot finding (`find_iconic_shots`) — OpenCV scans the film for
   close-up frontal faces, biased toward the climax (the back third), so the
   outro can rapid-fire the movie's most recognisable frames.

Shots can also be supplied by hand (timestamps typed in the web UI) and merged
with the auto-found ones.

Used two ways:
  - imported by main.py to assemble a recap render
  - shelled out to by the web UI:  python outro.py --scan <video> --music <track> --json
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── beat detection ───────────────────────────────────────────────────────────

def detect_beats(music_path: Path, max_time: float | None = None) -> list[float]:
    """Return the music's beat timestamps (seconds), optionally capped at max_time.

    Uses librosa's beat tracker. Imported lazily so the rest of the module (and
    the classic/narrated pipelines) don't pay librosa's heavy import cost.
    """
    import librosa  # heavy (numba/scipy) — only load it when a recap needs beats

    duration = None if max_time is None else max_time + 5.0
    y, sr = librosa.load(str(music_path), sr=22050, mono=True, duration=duration)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    if max_time is not None:
        beats = [b for b in beats if b <= max_time]
    return beats


def _nearest_beat(t: float, beats: list[float]) -> float | None:
    if not beats:
        return None
    return min(beats, key=lambda b: abs(b - t))


def snap_cuts_to_beats(
    ranges: list[tuple[float, float]],
    beats: list[float],
    tol: float = 0.5,
) -> list[tuple[float, float]]:
    """Nudge each clip's length so the CONCATENATED cut boundaries land on beats.

    The ranges are already snapped to complete-sentence boundaries by recap.py;
    here we only shift each clip's END by up to `tol` seconds so the running
    total (where one clip meets the next in the final video) coincides with a
    music beat. A sub-half-second nudge on a multi-second clip is inaudible but
    makes every cut hit the rhythm.
    """
    if not beats:
        return ranges
    snapped: list[tuple[float, float]] = []
    cum = 0.0
    for start, end in ranges:
        length = end - start
        target = cum + length
        nearest = _nearest_beat(target, beats)
        if nearest is not None and abs(nearest - target) <= tol:
            length = max(0.5, nearest - cum)
            end = start + length
        snapped.append((start, end))
        cum += length
    return snapped


# ── iconic-shot finding ──────────────────────────────────────────────────────

def find_iconic_shots(
    video_path: Path,
    count: int = 12,
    region: tuple[float, float] = (0.55, 0.98),
    sample_every: float = 1.5,
    min_gap: float = 2.0,
    max_samples: int = 500,
) -> list[dict]:
    """Scan `video_path` for close-up frontal faces and return the best shots.

    Samples a frame every `sample_every` seconds within `region` (fractions of
    the runtime — default the back ~45%, where the climax lives), scores each by
    the largest detected face's share of the frame, and returns up to `count`
    shots at least `min_gap` seconds apart.

    The step widens automatically if the region is long, so the scan never
    decodes more than `max_samples` frames (a full-movie random-seek scan would
    otherwise take minutes).

    Returns [{time, score}, ...] with time in seconds on the source timeline.
    """
    import cv2  # OpenCV import is non-trivial; keep it out of the classic path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for scanning: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if frame_count else 0.0
    if duration <= 0:
        cap.release()
        raise RuntimeError(f"Could not read duration from {video_path}")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    start_t = region[0] * duration
    end_t = region[1] * duration
    # Widen the step if needed so we decode at most max_samples frames.
    step = max(sample_every, (end_t - start_t) / max_samples)
    candidates: list[dict] = []

    t = start_t
    while t < end_t:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                              minSize=(int(w * 0.10), int(h * 0.10)))
            if len(faces):
                fw, fh = max(faces, key=lambda r: r[2] * r[3])[2:4]
                score = (fw * fh) / float(w * h)  # face's share of the frame
                candidates.append({"time": round(t, 2), "score": round(score, 4)})
        t += step

    cap.release()

    # Best faces first, but keep shots spread out in time (no two from the same
    # moment) so the outro cuts between different characters/frames.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    picked: list[dict] = []
    for cand in candidates:
        if all(abs(cand["time"] - p["time"]) >= min_gap for p in picked):
            picked.append(cand)
        if len(picked) >= count:
            break
    picked.sort(key=lambda c: c["time"])  # chronological for a nicer build order
    return picked


# ── outro assembly ───────────────────────────────────────────────────────────

def build_outro_flashes(
    beats: list[float],
    body_duration: float,
    outro_seconds: float,
    shot_times: list[float],
    video_duration: float,
    min_flash: float = 0.18,
) -> list[tuple[float, float]]:
    """Build the closing montage's source ranges, one flash per music beat.

    The outro occupies the final `outro_seconds` of the video, at
    [body_duration, body_duration + outro_seconds] on the OUTPUT timeline (which
    equals music time, since the bed starts at t=0). The music beats inside that
    window define the flash boundaries; each flash shows one iconic shot for the
    length of its beat interval, so the cuts land exactly on the beat.

    Returns source-video ranges to append after the body ranges. Falls back to
    evenly-spaced flashes if the music has no detectable beats in the window.
    """
    if not shot_times:
        return []

    window_start = body_duration
    window_end = body_duration + outro_seconds
    inner = [b for b in beats if window_start < b < window_end]
    boundaries = [window_start, *inner, window_end]

    # No beats landed in the window (very slow track / short outro) — fall back
    # to a fixed grid so the montage still fires rapidly.
    if len(boundaries) <= 2:
        n = max(3, int(outro_seconds / 0.5))
        step = outro_seconds / n
        boundaries = [window_start + i * step for i in range(n + 1)]

    flashes: list[tuple[float, float]] = []
    si = 0
    for i in range(len(boundaries) - 1):
        d = boundaries[i + 1] - boundaries[i]
        if d < min_flash:
            continue
        center = shot_times[si % len(shot_times)]
        si += 1
        s = max(0.0, center - d / 2)
        e = min(video_duration, s + d)
        s = max(0.0, e - d)  # keep full length if we clamped at the tail
        flashes.append((round(s, 3), round(e, 3)))
    return flashes


def _outro_durations(beats: list[float], body_duration: float, outro_seconds: float,
                     n_shots: int, lo: float = 0.5, hi: float = 1.0) -> list[float]:
    """Per-image hold times from the music beats, clamped to [lo, hi] seconds.

    Uses the beat intervals inside the outro window so image changes land on the
    beat, but never faster than `lo` (the old sub-0.5s flashes felt rushed) or
    slower than `hi`.
    """
    win = [b for b in beats if body_duration < b < body_duration + outro_seconds]
    bounds = [body_duration, *win, body_duration + outro_seconds]
    durs = [max(lo, min(hi, bounds[i + 1] - bounds[i])) for i in range(len(bounds) - 1)]
    durs = [d for d in durs if d >= lo * 0.6]
    if not durs:  # no beats in window — even grid
        n = max(1, int(outro_seconds / hi))
        durs = [outro_seconds / n] * n
    # Match the count to however many shots we have to show (cycle if needed).
    if n_shots and len(durs) < n_shots:
        durs = (durs * ((n_shots // len(durs)) + 1))[:n_shots]
    return durs


def build_animated_outro(
    video_path: Path,
    shot_times: list[float],
    durations: list[float],
    out_path: Path,
    size: tuple[int, int] = (1080, 1920),
    fps: int = 30,
    watermark_text: str | None = None,
    music_path: Path | None = None,
    music_start: float = 0.0,
    music_volume: float = 0.18,
    fade: float = 0.28,
) -> Path:
    """Build the animated outro: one enhanced face still per shot, each held for
    its beat-timed duration with a slow zoom-out + fade-in (Ken Burns), the
    watermark on top, and the music continuing underneath.

    Returns out_path (a self-contained h264/aac clip at `size`).
    """
    import subprocess
    import tempfile

    W, H = size
    enh = "eq=contrast=1.08:brightness=0.02:saturation=1.14,unsharp=5:5:0.9"
    # Cover the vertical frame, then a touch bigger so the zoom-out has room.
    cover = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"

    # Probe the source length once so we never seek past the end (a bad seek can
    # hang or emit an empty still, which used to crash the whole render).
    try:
        _p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=nw=1:nk=1", str(video_path)],
                            capture_output=True, text=True, check=True)
        vid_dur = float(_p.stdout.strip())
    except Exception:
        vid_dur = 0.0
    if vid_dur > 2:
        shot_times = [t for t in shot_times if 0 <= t <= vid_dur - 1.0]

    with tempfile.TemporaryDirectory(prefix="outro-") as tmp:
        tmp_dir = Path(tmp)
        clips: list[Path] = []
        for i, t in enumerate(shot_times):
            d = durations[i % len(durations)] if durations else 0.6
            total = max(2, int(round(d * fps)))
            still = tmp_dir / f"s{i:02d}.png"
            clip = tmp_dir / f"c{i:02d}.mp4"
            # 1) grab + enhance the still — skip this shot if it fails rather than
            # aborting the whole outro (and, upstream, the whole render).
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(t), "-i", str(video_path), "-frames:v", "1",
                     "-vf", f"{enh},{cover}", str(still)],
                    check=True, capture_output=True,
                )
                if not still.is_file() or still.stat().st_size == 0:
                    raise RuntimeError("empty still")
            except Exception as exc:
                print(f"      ! outro: skipping shot at {t:.0f}s ({exc})")
                continue
            # 2) animate: slow zoom-OUT (1.25x -> 1.0) + fade-in. Input framerate
            # set so the loop yields exactly `total` frames; zoompan d=1 emits one
            # output per input frame (d=total per frame explodes into a glitch).
            zoom = (f"zoompan=z='max(1.0,1.25-0.25*on/{total})':d=1:"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={fps},"
                    f"fade=t=in:st=0:d={fade},setsar=1,format=yuv420p")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loop", "1", "-framerate", str(fps), "-t", f"{d:.3f}",
                     "-i", str(still), "-vf", zoom, "-c:v", "libx264", "-crf", "18",
                     "-preset", "veryfast", "-pix_fmt", "yuv420p", str(clip)],
                    check=True, capture_output=True,
                )
                clips.append(clip)
            except Exception as exc:
                print(f"      ! outro: animate failed for shot at {t:.0f}s ({exc})")
                continue

        if not clips:
            raise RuntimeError("no outro shots to animate")

        # 3) concat the animated stills
        concat_list = tmp_dir / "list.txt"
        concat_list.write_text("\n".join(f"file '{c.as_posix()}'" for c in clips), encoding="utf-8")
        silent = tmp_dir / "silent.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(silent)],
            check=True, capture_output=True,
        )
        total_dur = sum((durations[i % len(durations)] if durations else 0.6)
                        for i in range(len(clips)))

        # 4) watermark on top + music underneath
        vf = []
        if watermark_text:
            safe = watermark_text.replace("'", "").replace(":", "")
            vf.append(f"drawtext=text='{safe}':fontsize=40:fontcolor=white@0.85:"
                      f"x=(w-tw)/2:y=h-th-70:box=1:boxcolor=black@0.35:boxborderw=12")
        vf_arg = ",".join(vf) if vf else "null"

        cmd = ["ffmpeg", "-y", "-i", str(silent)]
        if music_path and Path(music_path).is_file():
            cmd += ["-ss", str(music_start), "-i", str(music_path),
                    "-filter_complex",
                    f"[0:v]{vf_arg}[v];[1:a]atrim=0:{total_dur:.3f},asetpts=PTS-STARTPTS,"
                    f"volume={music_volume},aresample=48000,aformat=channel_layouts=stereo[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", str(out_path)]
        else:
            cmd += ["-vf", vf_arg, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", str(out_path)]
        subprocess.run(cmd, check=True, capture_output=True)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a movie for iconic face shots and a music beat grid.")
    parser.add_argument("--scan", required=True, help="Path to the source video to scan for faces")
    parser.add_argument("--music", help="Path to the music track (for the beat grid)")
    parser.add_argument("--count", type=int, default=12, help="How many iconic shots to return (default: 12)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args()

    video = Path(args.scan)
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        sys.exit(1)

    shots = find_iconic_shots(video, count=args.count)
    beats = detect_beats(Path(args.music)) if args.music and Path(args.music).is_file() else []

    if args.json:
        print(json.dumps({"shots": shots, "beats": beats}, ensure_ascii=False))
        return

    print(f"{len(shots)} iconic shot(s):")
    for s in shots:
        print(f"  {s['time']:8.2f}s  (face fills {s['score'] * 100:.0f}% of frame)")
    print(f"\n{len(beats)} music beat(s) detected." if beats else "\nNo music track scanned.")


if __name__ == "__main__":
    main()
