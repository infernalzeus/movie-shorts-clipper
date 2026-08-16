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
    sample_every: float = 1.0,
    min_gap: float = 2.0,
    max_samples: int = 500,
    windows: list[tuple[float, float]] | None = None,
    min_score: float = 0.0,
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

    # Sample inside the given windows (the recap beats) if provided, else across
    # `region`. This makes the outro's faces come from the story just shown.
    spans = windows if windows else [(region[0] * duration, region[1] * duration)]
    spans = [(max(0.0, s), min(duration, e)) for s, e in spans if e > s]
    total_span = sum(e - s for s, e in spans) or 1.0
    step = max(sample_every, total_span / max_samples)
    sample_times: list[float] = []
    for s, e in spans:
        t = s
        while t < e:
            sample_times.append(t)
            t += step

    candidates: list[dict] = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # minNeighbors 8 + larger minSize => only high-confidence close-up faces.
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8,
                                          minSize=(int(w * 0.14), int(h * 0.14)))
        if len(faces):
            fx0, fy0, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            score = (fw * fh) / float(w * h)  # face's share of the frame
            if len(faces) == 1:               # prefer clean SINGLE-face frames
                score *= 1.6
            if score >= min_score:
                candidates.append({
                    "time": round(t, 2), "score": round(score, 4),
                    "fx": round((fx0 + fw / 2) / w, 4),   # face centre, normalised
                    "fy": round((fy0 + fh / 2) / h, 4),
                    "faces": int(len(faces)),
                })

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
                     n_shots: int, lo: float = 0.8, hi: float = 1.6) -> list[float]:
    """Hold time for each outro clip: spread `outro_seconds` across `n_shots`
    evenly, clamped to [lo, hi]. Kept long enough that the clips aren't squished
    (the earlier beat-synced version packed too many, too fast).
    """
    if n_shots <= 0:
        return []
    per = max(lo, min(hi, outro_seconds / n_shots))
    return [per] * n_shots


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
    movie_name: str | None = None,
    shot_faces: list | None = None,
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
    # Pair each shot time with its face's horizontal centre (0..1), if known.
    faces_in = shot_faces if (shot_faces and len(shot_faces) == len(shot_times)) else [None] * len(shot_times)
    pairs = list(zip(shot_times, faces_in))
    if vid_dur > 2:
        pairs = [(t, fx) for t, fx in pairs if 0 <= t <= vid_dur - 1.0]

    with tempfile.TemporaryDirectory(prefix="outro-") as tmp:
        tmp_dir = Path(tmp)
        clips: list[Path] = []
        for i, (t, fx) in enumerate(pairs):
            d = durations[i % len(durations)] if durations else 0.6
            clip = tmp_dir / f"c{i:02d}.mp4"
            # Frame a full-height PORTRAIT window centred on the face (so the face
            # is centred and never cut off the sides); fall back to a centre crop.
            if fx is not None:
                cw = f"ih*{W}/{H}"
                cover_i = (f"crop=w='{cw}':h=ih:"
                           f"x='min(max(0\\,{fx}*iw-({cw})/2)\\,iw-({cw}))':y=0,scale={W}:{H}")
            else:
                cover_i = cover
            # Use a short REAL clip of footage (not a still), slowed to fill the
            # display duration `d` (so it's moving slow-motion), with a slow
            # zoom-out (crop grows from 0.82 -> full over d) + fade-in. Skip a shot
            # on failure rather than aborting the outro (or the whole render).
            src_len = max(0.2, min(d, 0.55))     # footage grabbed; stretched to d
            slow = max(1.0, d / src_len)          # >1 = slow motion
            start = max(0.0, t - src_len / 2)
            total = max(2, int(round(d * fps)))
            # Speed RAMP fast->slow: setpts maps input time T via a convex power
            # curve (p=1.8) so early frames are compressed (fast) and later frames
            # spread out (slow) across the display duration `d`. fps fills the
            # ramped frames, zoompan does the slow zoom-out. `-t` goes BEFORE `-i`
            # (input duration) so setpts can stretch it past the source length.
            vf = (f"{enh},{cover_i},setpts=({d:.3f}*pow(T/{src_len:.3f}\\,1.8))/TB,fps={fps},"
                  f"zoompan=z='max(1.0,1.25-0.25*on/{total})':d=1:"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={fps},"
                  f"fade=t=in:st=0:d={fade},setsar=1,format=yuv420p")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{src_len:.3f}",
                     "-i", str(video_path), "-an", "-vf", vf, "-c:v", "libx264",
                     "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(clip)],
                    check=True, capture_output=True,
                )
                if clip.is_file() and clip.stat().st_size > 0:
                    clips.append(clip)
                else:
                    raise RuntimeError("empty clip")
            except Exception as exc:
                print(f"      ! outro: skipping shot at {t:.0f}s ({exc})")
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
        # Actual length of the concatenated clips (the speed-ramp makes each a bit
        # longer than its target `d`, so the intended sum is wrong for the xfade).
        try:
            _sp = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                  "format=duration", "-of", "default=nw=1:nk=1", str(silent)],
                                 capture_output=True, text=True, check=True)
            total_dur = float(_sp.stdout.strip())
        except Exception:
            total_dur = sum((durations[i % len(durations)] if durations else 0.6)
                            for i in range(len(clips)))

        # 4) Titles (movie name top + MV EDITS bottom, black text with a white
        #    glow, fading in), then an expanding-circle BLACKOUT to end on black.
        # Pick a fancier font (first that exists): Cinzel/Trajan-like serifs read
        # as "cinematic"; fall back through elegant Windows serifs to Arial Bold.
        _font = next((n for n in ("Cinzel-Bold.ttf", "CinzelDecorative-Bold.ttf",
                                  "PlayfairDisplay-Bold.ttf", "constanb.ttf", "georgiab.ttf",
                                  "BOD_B.TTF", "impact.ttf", "arialbd.ttf")
                      if (Path("C:/Windows/Fonts") / n).is_file()), None)
        ff = f"fontfile='C\\:/Windows/Fonts/{_font}':" if _font else ""
        fin = f"alpha='if(lt(t\\,0.5)\\,t/0.5\\,1)'"   # 0.5s fade-in

        def _txt(text: str, y: str, fs: int, color: str) -> str:
            safe = (text or "").replace("'", "").replace(":", " ").replace("\\", "")
            return (f"drawtext={ff}text='{safe}':fontcolor={color}:fontsize={fs}:"
                    f"x=(w-tw)/2:y={y}:{fin}")

        # (text, y-expr, fontsize) for each title.
        specs = []
        if movie_name:
            specs.append((movie_name, "110", 54))
        if watermark_text:
            specs.append((watermark_text, "h-th-170", 66))
        white = ",".join(_txt(t, y, fs, "white") for t, y, fs in specs)
        black = ",".join(_txt(t, y, fs, "black") for t, y, fs in specs)

        # Blackout that radiates FROM the watermark (bottom-centre) with a SOFT,
        # feathered edge (a glow-spread, not a hard-chopped circle): geq multiplies
        # each pixel by a smooth 1->0 falloff over `fea` px at the growing radius.
        import math
        bd = 1.4
        cx, cy = W // 2, H - 210
        maxR = int(math.hypot(max(cx, W - cx), max(cy, H - cy))) + 80
        fea = 130
        new_total = total_dur + bd
        R = f"{maxR}*clip((T-{total_dur:.3f})/{bd}\\,0\\,1)"
        fexpr = f"clip(({R}-hypot(X-{cx}\\,Y-{cy}))/{fea}\\,0\\,1)"
        geq = (f"format=gbrp,geq=r='r(X\\,Y)*(1-{fexpr})':"
               f"g='g(X\\,Y)*(1-{fexpr})':b='b(X\\,Y)*(1-{fexpr})',format=yuv420p")

        # Glow = white copy of the text blurred into a HALO (not a border), screen-
        # blended onto the video, with the crisp black text drawn on top.
        cmd = ["ffmpeg", "-y", "-i", str(silent)]
        if specs:
            cmd += ["-f", "lavfi", "-i", f"color=black:s={W}x{H}:d={total_dur:.3f}:r={fps}"]
            # Blend in RGB (gbrp), NOT yuv — screen-blending yuv chroma shifts the
            # colours (that was the purple cast). Everything stays gbrp until the
            # final format=yuv420p inside `geq`.
            vchain = (f"[1:v]{white},gblur=sigma=9,format=gbrp[halo];"
                      f"[0:v]fps={fps},setsar=1,format=gbrp[base];"
                      f"[base][halo]blend=all_mode=screen[glowed];"
                      f"[glowed]{black},tpad=stop_mode=clone:stop_duration={bd:.3f},{geq}[v]")
            music_idx = 2
        else:
            vchain = (f"[0:v]fps={fps},setsar=1,"
                      f"tpad=stop_mode=clone:stop_duration={bd:.3f},{geq}[v]")
            music_idx = 1

        if music_path and Path(music_path).is_file():
            cmd += ["-ss", str(music_start), "-i", str(music_path),
                    "-filter_complex",
                    vchain + f";[{music_idx}:a]atrim=0:{new_total:.3f},asetpts=PTS-STARTPTS,"
                    f"volume={music_volume},aresample=48000,aformat=channel_layouts=stereo,"
                    f"afade=t=out:st={new_total - 0.6:.3f}:d=0.6[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", str(out_path)]
        else:
            cmd += ["-filter_complex", vchain, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
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
