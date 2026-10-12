"""
Web UI for movie-shorts-clipper.

Standalone app — talks to src/main.py exactly like any other caller (e.g. the
Agent CORE Telegram bot) by shelling out to it as a subprocess. Does not
import from or modify the pipeline itself, so the CLI contract stays intact
for other callers.

Run:  python web/app.py
Env:  CLIPPER_WEB_HOST (default 0.0.0.0), CLIPPER_WEB_PORT (default 8081),
      CLIPPER_BASE_PATH (default "" — set by Agent Hub when reverse-proxied)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import urllib.parse
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web, WSMsgType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
MAIN_SCRIPT  = SRC_DIR / "main.py"
OUTPUT_DIR   = PROJECT_ROOT / "output"
AUDIO_DIR    = PROJECT_ROOT / "audio"

HOST = os.getenv("CLIPPER_WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("CLIPPER_WEB_PORT", "8081"))
# Empty by default (standalone, rooted at "/"). Set by Agent Hub when it spawns this
# app behind its reverse proxy, so client-facing URLs resolve under /app/movie-clipper/.
BASE_PATH = os.getenv("CLIPPER_BASE_PATH", "").rstrip("/")

_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a"}
STEP_RE = re.compile(r"^\[(\d+)/(\d+)\]\s*(.*)$")

SCENES_SCRIPT = SRC_DIR / "scenes.py"
RECAP_SCRIPT = SRC_DIR / "recap.py"
EDITORIAL_SCRIPT = SRC_DIR / "editorial.py"
THUMBNAIL_SCRIPT = SRC_DIR / "thumbnail.py"
METADATA_SCRIPT = SRC_DIR / "metadata.py"

# Every full run needs the local Ollama daemon (metadata step always uses it,
# narration too; cloud models still go through the local daemon). Checking up
# front — and trying to start it — beats a traceback 10 minutes into a render.
OLLAMA_VERSION_URL = "http://127.0.0.1:11434/api/version"


async def _ollama_up(timeout: float = 1.5) -> bool:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(OLLAMA_VERSION_URL, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status == 200
    except Exception:
        return False


_ollama_spawn_lock = asyncio.Lock()


def _spawn_ollama() -> str | None:
    """Best-effort Ollama launch. Returns an error message, or None if spawned.

    Windows flags matter here: DETACHED_PROCESS would strip the daemon of any
    console, making every model-runner child it spawns allocate a fresh
    VISIBLE console — glitchy cmd windows popping up on the desktop.
    CREATE_NO_WINDOW gives it a hidden console the children inherit instead.
    Prefer the Ollama desktop app when installed (it manages its own daemon
    and tray icon — the way Ollama normally runs on this machine).
    """
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        app_exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
        try:
            desktop_available = app_exe.is_file()
        except OSError:
            desktop_available = False
        if desktop_available:
            try:
                subprocess.Popen(
                    [str(app_exe)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=flags,
                )
                return None
            except Exception:
                pass  # fall through to the CLI
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        return None
    except FileNotFoundError:
        return ("Ollama isn't running and the 'ollama' command wasn't found — "
                "start the Ollama app, then retry.")
    except Exception as exc:
        return f"Ollama isn't running and starting it failed ({exc}) — start it manually, then retry."


async def _ensure_ollama() -> str | None:
    """Return None if Ollama is reachable (starting it if needed), else a
    user-facing error message."""
    if await _ollama_up():
        return None
    # Lock so concurrent requests can't race into spawning multiple daemons.
    async with _ollama_spawn_lock:
        if await _ollama_up():
            return None
        err = _spawn_ollama()
        if err:
            return err
        for _ in range(20):  # up to ~10s for the daemon to come up
            await asyncio.sleep(0.5)
            if await _ollama_up():
                return None
    return ("Ollama isn't running (localhost:11434 unreachable) and auto-starting "
            "it didn't bring it up — start Ollama manually, then retry.")


def _clean_path(value: str | None) -> str:
    """Strip whitespace and any surrounding quotes from a pasted file path.

    Windows' 'Copy as path' wraps the path in double quotes, and users often
    paste those in — accept the path anyway instead of reporting 'not found'.

    Also accept URL-percent-encoded pastes: some iOS paste targets deliver a
    copied path as a URL flavor ('\\' as %5C, ' ' as %20) while others paste it
    plain. A real Windows path never contains %5C/%20/%3A literally, so seeing
    one means the string is encoded — decode it.
    """
    s = (value or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("\"", "'"):
        s = s[1:-1]
    s = s.strip()
    if re.search(r"%(5C|20|3A|2F)", s, re.IGNORECASE):
        try:
            s = urllib.parse.unquote(s)
        except Exception:
            pass
    if s.lower().startswith("file:///"):
        s = urllib.parse.unquote(s[len("file:///"):]).replace("/", "\\")
    return s.strip()


def _find_srt_for_video(video_path: Path) -> Path | None:
    """Same heuristic as src/subtitles.py (duplicated so the web app keeps
    shelling out instead of importing the pipeline): same-stem .srt first,
    then any .srt in the folder preferring English-looking names."""
    exact = video_path.with_suffix(".srt")
    if exact.is_file():
        return exact
    candidates = sorted(video_path.parent.glob("*.srt"))
    if not candidates:
        return None
    for cand in candidates:
        if re.search(r"\b(english|eng|en)\b", cand.stem, re.IGNORECASE):
            return cand
    return candidates[0]


# ── single-run state (personal-use scope — one clip at a time) ────────────────

class RunState:
    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.lines: list[str] = []
        self.step: int = 0
        # Set per-run from _expected_steps() and corrected by the first '[n/N]'
        # line main.py prints. The old hardcoded 6 predated narration and was
        # simply wrong: a narrated run with thumbnail baking is 9.
        self.total_steps: int = 0
        self.step_text: str = ""
        self.done: bool = False
        self.ok: bool = False
        self.result: dict = {}
        self.subscribers: set[web.WebSocketResponse] = set()

    def reset(self) -> None:
        self.proc = None
        self.lines = []
        self.step = 0
        self.total_steps = 0
        self.step_text = ""
        self.done = False
        self.ok = False
        self.result = {}


run_state = RunState()

# Phase-1 "prepared" clip (cut, awaiting a thumbnail pick before the full render).
# video/ranges record what the cut was made FROM, so /run can refuse to reuse it
# for a different request instead of rendering the old clip's footage.
_prepared: dict = {"clip_dir": None, "raw_clip_url": None, "video": None, "ranges": None}
_prepare_lock = asyncio.Lock()


async def _broadcast(payload: dict) -> None:
    dead = []
    for ws in run_state.subscribers:
        try:
            await ws.send_str(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        run_state.subscribers.discard(ws)


def _rel_output_url(path_str: str) -> str | None:
    """Map an absolute path the CLI printed back to a /output/... URL, if it's inside OUTPUT_DIR."""
    try:
        rel = Path(path_str).resolve().relative_to(OUTPUT_DIR.resolve())
    except (ValueError, OSError):
        return None
    return BASE_PATH + "/output/" + rel.as_posix()


async def _stream_run(args: list[str]) -> None:
    cmd = [sys.executable, str(MAIN_SCRIPT)] + args
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Unbuffered: stdout is a pipe here, and Python block-buffers pipes (~8KB).
    # Without this the '[n/N]' progress lines sit in the buffer and flush in one
    # burst at the end, so the UI's step bar never moves during the run.
    env["PYTHONUNBUFFERED"] = "1"

    # New process group on Windows so we can kill the whole ffmpeg/edge-tts tree
    # on shutdown (see _terminate_run) rather than leaving orphans behind.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SRC_DIR),
        env=env,
        creationflags=creationflags,
    )
    run_state.proc = proc

    async def _pump() -> None:
        assert proc.stdout
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            run_state.lines.append(line)

            m = STEP_RE.match(line)
            if m:
                run_state.step = int(m.group(1))
                run_state.total_steps = int(m.group(2))
                run_state.step_text = m.group(3).strip().rstrip(".…")

            await _broadcast({
                "type": "line", "text": line,
                "step": run_state.step, "total": run_state.total_steps,
                "step_text": run_state.step_text,
            })

    # main.py's exit — not stdout EOF — is the authoritative completion signal.
    # A stray grandchild (a lingering ffmpeg, an edge-tts SSL transport) can hold
    # the stdout pipe open after main.py has finished and exited, which would
    # otherwise leave `async for ... in proc.stdout` blocked forever and the UI
    # stuck on "running". So we wait on the process, then drain output briefly.
    pump = asyncio.create_task(_pump())
    rc = await proc.wait()
    try:
        await asyncio.wait_for(pump, timeout=5)
    except asyncio.TimeoutError:
        pump.cancel()

    run_state.done = True
    run_state.ok = (rc == 0)

    if run_state.ok:
        result: dict = {}
        for line in run_state.lines:
            if "Final video:" in line:
                p = line.split("Final video:")[-1].strip()
                result["video_path"] = p
                result["video_url"] = _rel_output_url(p)
            elif "Title:" in line and "title" not in result:
                result["title"] = line.split("Title:")[-1].strip()
            elif "Movie:" in line and "movie" not in result:
                result["movie"] = line.split("Movie:")[-1].strip()
            elif "Metadata:" in line:
                p = line.split("Metadata:")[-1].strip()
                result["metadata_url"] = _rel_output_url(p)
            elif "Description:" in line:
                p = line.split("Description:")[-1].strip()
                result["description_url"] = _rel_output_url(p)
            elif "Thumbnail:" in line:
                p = line.split("Thumbnail:")[-1].strip()
                result["thumbnail_url"] = _rel_output_url(p)
            elif "Narration:" in line:
                p = line.split("Narration:")[-1].strip()
                result["narration_url"] = _rel_output_url(p)
            elif "Raw clip:" in line:
                p = line.split("Raw clip:")[-1].strip()
                result["raw_clip_path"] = p
                result["raw_clip_url"] = _rel_output_url(p)
        run_state.result = result

    await _broadcast({"type": "done", "ok": run_state.ok, "result": run_state.result})


# ── routes ──────────────────────────────────────────────────────────────────

routes = web.RouteTableDef()


@routes.get("/")
async def index(request: web.Request) -> web.Response:
    return web.Response(text=_render_html(), content_type="text/html", charset="utf-8")


@routes.get("/favicon.png")
async def favicon_png(request: web.Request) -> web.Response:
    path = PROJECT_ROOT / "web" / "favicon.png"
    if not path.exists():
        raise web.HTTPNotFound()
    return web.Response(body=path.read_bytes(), content_type="image/png")


@routes.get("/favicon.svg")
async def favicon_svg(request: web.Request) -> web.Response:
    return web.Response(text=_render_favicon_svg(), content_type="image/svg+xml", charset="utf-8")


@routes.get("/audio-files")
async def audio_files(request: web.Request) -> web.Response:
    files = []
    if AUDIO_DIR.is_dir():
        files = sorted(f.name for f in AUDIO_DIR.iterdir() if f.suffix.lower() in _AUDIO_EXTS)
    return web.json_response(files)


@routes.get("/enhance-presets")
async def enhance_presets(request: web.Request) -> web.Response:
    """The thumbnail enhancement presets, straight from thumbnail.py, so the UI's
    sliders can't drift from the values the pipeline actually applies."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(THUMBNAIL_SCRIPT), "--list-presets",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        cwd=str(SRC_DIR),
    )
    stdout, _ = await proc.communicate()
    try:
        return web.json_response(json.loads(stdout.decode(errors="replace").strip()))
    except (json.JSONDecodeError, AttributeError):
        raise web.HTTPInternalServerError(text="could not read enhancement presets")


@routes.get("/status")
async def status(request: web.Request) -> web.Response:
    busy = run_state.proc is not None and run_state.proc.returncode is None
    return web.json_response({"busy": busy})


@routes.post("/probe-video")
async def probe_video(request: web.Request) -> web.Response:
    """Check a movie path: does it exist, and does it have a subtitle file next to it?"""
    data = await request.json()
    video = _clean_path(data.get("video"))
    path = Path(video) if video else None
    if not path or not path.is_file():
        return web.json_response({"exists": False, "srt": None})
    srt = _find_srt_for_video(path)
    # Runtime (seconds) so the UI can bound the "summarize this span" slider.
    duration = None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        duration = round(float(r.stdout.strip()), 1)
    except Exception:
        duration = None
    return web.json_response({"exists": True, "srt": str(srt) if srt else None, "duration": duration})


_suggest_lock = asyncio.Lock()


@routes.post("/suggest-scenes")
async def suggest_scenes(request: web.Request) -> web.Response:
    """Run src/scenes.py against the movie's SRT and return suggested clip ranges."""
    data = await request.json()
    srt = (data.get("srt") or "").strip()
    if not srt or not Path(srt).is_file():
        raise web.HTTPBadRequest(text="SRT file not found")

    args = [sys.executable, str(SCENES_SCRIPT), "--srt", srt, "--json"]
    if data.get("title"):
        args += ["--title", str(data["title"])]
    if data.get("model"):
        args += ["--model", str(data["model"])]
    args += ["--count", str(int(data.get("count") or 5))]
    # Scene length is shape-driven now: a 9:16 Short wants 20-60s, a 16:9
    # long-form video wants minutes. The client sends the window it wants.
    for key, flag in (("min_seconds", "--min-seconds"), ("max_seconds", "--max-seconds")):
        if data.get(key) is not None:
            try:
                args += [flag, f"{float(data[key]):g}"]
            except (TypeError, ValueError):
                raise web.HTTPBadRequest(text=f"Invalid {key}")

    if _suggest_lock.locked():
        raise web.HTTPConflict(text="A scene suggestion is already running")
    ollama_err = await _ensure_ollama()
    if ollama_err:
        raise web.HTTPServiceUnavailable(text=ollama_err)
    async with _suggest_lock:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(SRC_DIR),
            env=env,
        )
        stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        detail = (stderr or stdout or b"").decode(errors="replace")[-800:]
        raise web.HTTPInternalServerError(text=f"scene suggestion failed:\n{detail}")
    try:
        payload = json.loads(stdout.decode(errors="replace").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise web.HTTPInternalServerError(text="scene suggestion returned unparseable output")
    return web.json_response(payload)


_recap_lock = asyncio.Lock()


@routes.post("/build-recap")
async def build_recap(request: web.Request) -> web.Response:
    """Run src/recap.py against the movie's SRT and return an ordered chain of
    plot-recap beats (with a ready-to-use combined ranges string)."""
    data = await request.json()
    srt = (data.get("srt") or "").strip()
    if not srt or not Path(srt).is_file():
        raise web.HTTPBadRequest(text="SRT file not found")

    args = [sys.executable, str(RECAP_SCRIPT), "--srt", srt, "--json"]
    if data.get("title"):
        args += ["--title", str(data["title"])]
    if data.get("model"):
        args += ["--model", str(data["model"])]
    if data.get("target_seconds"):
        args += ["--target-seconds", str(data["target_seconds"])]

    if _recap_lock.locked():
        raise web.HTTPConflict(text="A recap build is already running")
    ollama_err = await _ensure_ollama()
    if ollama_err:
        raise web.HTTPServiceUnavailable(text=ollama_err)
    async with _recap_lock:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(SRC_DIR),
            env=env,
        )
        stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        detail = (stderr or stdout or b"").decode(errors="replace")[-800:]
        raise web.HTTPInternalServerError(text=f"recap build failed:\n{detail}")
    try:
        payload = json.loads(stdout.decode(errors="replace").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise web.HTTPInternalServerError(text="recap build returned unparseable output")
    return web.json_response(payload)


_titles_lock = asyncio.Lock()


@routes.post("/suggest-titles")
async def suggest_titles(request: web.Request) -> web.Response:
    """Propose on-screen title options for the clip, at thumbnail-pick time.

    Reads the dialogue straight out of the subtitle file for the chosen ranges,
    so this is a single LLM call rather than a transcription — fast enough to sit
    in front of the render, where the choice can still be applied.
    """
    data = await request.json()
    # Only use a subtitle file the caller actually SELECTED for this clip — never
    # fall back to some other .srt in the folder. When none is selected we pull
    # the dialogue from the already-cut clip with Whisper instead, so the titles
    # describe THIS clip, not whatever subtitle happened to be lying around.
    srt = (data.get("srt") or "").strip()
    ranges = (data.get("ranges") or "").strip()
    raw_clip = _resolve_output(data.get("raw_clip") or "")

    args = [
        sys.executable, str(METADATA_SCRIPT),
        "--titles", str(int(data.get("count") or 5)),
    ]
    if srt and Path(srt).is_file():
        if not ranges:
            raise web.HTTPBadRequest(text="At least one range is required")
        args += ["--srt", srt, "--ranges", ranges]
    elif raw_clip is not None and raw_clip.is_file():
        args += ["--clip", str(raw_clip),
                 "--whisper-model", str(data.get("whisper_model") or "medium"),
                 "--language", str(data.get("language") or "en")]
    else:
        raise web.HTTPBadRequest(
            text="No subtitle selected and the cut clip isn't available yet — "
                 "cut the clip first (it's transcribed for the title options).")
    if data.get("source_title"):
        args += ["--source-title", str(data["source_title"])]
    if data.get("model"):
        args += ["--model", str(data["model"])]

    if _titles_lock.locked():
        raise web.HTTPConflict(text="Title options are already being generated")
    ollama_err = await _ensure_ollama()
    if ollama_err:
        raise web.HTTPServiceUnavailable(text=ollama_err)
    async with _titles_lock:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(SRC_DIR), env=env,
        )
        stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        detail = (stderr or stdout or b"").decode(errors="replace")[-800:]
        raise web.HTTPInternalServerError(text=f"title suggestion failed:\n{detail}")
    try:
        payload = json.loads(stdout.decode(errors="replace").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise web.HTTPInternalServerError(text="title suggestion returned unparseable output")
    return web.json_response(payload)


def _resolve_output(url_or_rel: str) -> Path | None:
    """Map a served /output/... URL (or bare relative path) back to a disk path,
    guarding against escaping OUTPUT_DIR."""
    s = (url_or_rel or "").strip()
    marker = "/output/"
    if marker in s:
        s = s.split(marker, 1)[1]
    s = s.lstrip("/")
    if not s:
        return None
    full = (OUTPUT_DIR / s).resolve()
    out_root = OUTPUT_DIR.resolve()
    if full != out_root and out_root not in full.parents:
        return None
    return full


def _resolve_prepared_dir(value) -> Path | None:
    """Validate a prepared clip dir: must sit inside OUTPUT_DIR and still hold
    the prepared cut. Returns the resolved Path or None."""
    if not value:
        return None
    try:
        full = Path(str(value)).resolve()
    except OSError:
        return None
    out_root = OUTPUT_DIR.resolve()
    if full != out_root and out_root not in full.parents:
        return None
    if not (full / "clip_raw.mp4").is_file():
        return None
    return full


def _prepared_ranges_for(prep: Path, client_claim) -> str | None:
    """The ranges string the prepared cut in `prep` was actually cut from.

    Prefers this process's own record of the prepare; falls back to what the
    client says it prepared with (the record is lost on an app restart, but the
    clip dir on disk isn't). None means unknowable — main.py's prepare.json
    check is the backstop in that case.
    """
    if _prepared.get("clip_dir") and _resolve_prepared_dir(_prepared["clip_dir"]) == prep:
        return _prepared.get("ranges")
    claim = (client_claim or "").strip()
    return claim or None


_thumb_lock = asyncio.Lock()


@routes.post("/make-thumbnail")
async def make_thumbnail_route(request: web.Request) -> web.Response:
    """Regenerate the thumbnail from a user-picked frame of the raw clip.

    Body: {video: "<served /output url of the raw clip>", at: <seconds>,
           frame: "<data-URL jpeg of the exact displayed frame>",
           title: "...", movie: "..."}. Writes thumbnail.jpg beside the clip.
    """
    data = await request.json()
    src = _resolve_output(data.get("video") or "")
    if src is None or not src.is_file():
        raise web.HTTPBadRequest(text="Source clip not found")

    out_path = src.parent / "thumbnail.jpg"
    # The browser sends the exact pixels it is displaying (canvas capture) —
    # Safari snaps its scrub preview to keyframes, so a plain timestamp grab
    # can land ~half a second from the frame the user was looking at. When
    # the original movie path + mapped source time are also provided, the
    # capture is used to LOCATE that frame in the source and the pixels are
    # grabbed there at full quality (thumbnail.py --match-image; it falls
    # back to the capture itself if nothing matches).
    frame_path: Path | None = None
    frame_b64 = data.get("frame") or ""
    if frame_b64:
        try:
            frame_path = src.parent / "_picked_frame.jpg"
            frame_path.write_bytes(base64.b64decode(frame_b64.split(",", 1)[-1]))
        except Exception:
            frame_path = None

    source_video = _clean_path(data.get("source_video"))
    source_at = data.get("source_at")
    at = data.get("at")

    grab_video = str(src)
    frame_args: list[str] = []
    if frame_path is not None and source_video and Path(source_video).is_file() and source_at is not None:
        try:
            frame_args = ["--match-image", str(frame_path), "--at", f"{float(source_at):.3f}"]
            grab_video = source_video
        except (TypeError, ValueError):
            frame_args = ["--image", str(frame_path)]
    elif frame_path is not None:
        frame_args = ["--image", str(frame_path)]
    elif at is not None:
        try:
            frame_args = ["--at", f"{float(at):.3f}"]
        except (TypeError, ValueError):
            raise web.HTTPBadRequest(text="Invalid timestamp")

    args = [
        sys.executable, str(THUMBNAIL_SCRIPT),
        "--video", grab_video,
        "--out", str(out_path),
    ] + frame_args
    pan = data.get("pan")
    if pan is not None:
        try:
            args += ["--pan", f"{min(1.0, max(0.0, float(pan))):.4f}"]
        except (TypeError, ValueError):
            raise web.HTTPBadRequest(text="Invalid pan")
    # Canvas size follows the output shape — a 16:9 long-form upload needs a
    # 1920x1080 thumbnail, not the tall Shorts one.
    if data.get("width") and data.get("height"):
        try:
            args += ["--width", str(int(data["width"])), "--height", str(int(data["height"]))]
        except (TypeError, ValueError):
            raise web.HTTPBadRequest(text="Invalid thumbnail size")
    if data.get("title"):
        args += ["--title", str(data["title"])]
    if data.get("movie"):
        args += ["--movie", str(data["movie"])]
    if data.get("font"):
        args += ["--font", str(data["font"])]
    if data.get("color"):
        args += ["--color", str(data["color"])]
    if data.get("enhance"):
        args += ["--enhance", str(data["enhance"])]
    # Per-knob overrides on top of the preset. Sent only when the user has opened
    # the advanced sliders, so a plain preset pick stays a plain preset pick.
    for key, flag in (("brightness", "--brightness"), ("contrast", "--contrast"),
                      ("saturation", "--saturation"), ("sharpness", "--sharpness")):
        if data.get(key) is not None:
            try:
                args += [flag, f"{float(data[key]):.3f}"]
            except (TypeError, ValueError):
                raise web.HTTPBadRequest(text=f"Invalid {key}")

    if _thumb_lock.locked():
        raise web.HTTPConflict(text="A thumbnail is already being generated")
    async with _thumb_lock:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(SRC_DIR),
            env=env,
        )
        stdout, stderr = await proc.communicate()

    if frame_path is not None:
        frame_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        detail = (stderr or stdout or b"").decode(errors="replace")[-800:]
        raise web.HTTPInternalServerError(text=f"thumbnail generation failed:\n{detail}")

    url = _rel_output_url(str(out_path))
    # Cache-buster so the <img> refreshes after regeneration.
    if url:
        url += f"?t={int(asyncio.get_event_loop().time() * 1000)}"
    return web.json_response({"ok": True, "thumbnail_url": url})


@routes.post("/prepare")
async def prepare(request: web.Request) -> web.Response:
    """Phase 1: cut the raw clip so the user can pick a thumbnail from it before
    the (slow) full render. Returns the raw clip's URL; the clip dir is held
    server-side and reused by /run when the user renders."""
    if run_state.proc is not None and run_state.proc.returncode is None:
        raise web.HTTPConflict(text="A clip is already running")

    data = await request.json()
    video = _clean_path(data.get("video"))
    ranges = (data.get("ranges") or "").strip()
    if not video or not Path(video).is_file():
        raise web.HTTPBadRequest(text="Video file not found")
    if not ranges:
        raise web.HTTPBadRequest(text="At least one range is required")

    args = [sys.executable, str(MAIN_SCRIPT), "--video", video, "--ranges", ranges, "--prepare-only"]
    if data.get("source_title"):
        args += ["--source-title", str(data["source_title"])]
    # Phase 1 owns the cut, so it must snap boundaries the SAME way phase 2 will —
    # pass the toggle through or a snapped prepared cut won't match the captions.
    if data.get("no_snap_speech"):
        args += ["--no-snap-speech"]
    # Recap cuts its body on the music's beats, so phase 1 has to know the mode
    # and the track — a prepared cut that ignored them would be re-cut in phase 2
    # and take the hand-picked thumbnail down with it.
    if (data.get("selection") or "") == "spine" or (data.get("mode") or "") == "recap":
        args += ["--selection", "spine", "--voice", str(data.get("voice") or "music")]
        if data.get("srt") and Path(str(data["srt"])).is_file():
            args += ["--srt", str(data["srt"])]
        bg_mode = data.get("bg_mode") or "none"
        if bg_mode == "none":
            args += ["--no-bg-music"]
        elif bg_mode == "file" and data.get("bg_music"):
            bg_path = AUDIO_DIR / str(data["bg_music"])
            if bg_path.is_file():
                args += ["--bg-music", str(bg_path)]
        # bg_mode == "random" falls through: main.py rolls the track, prints it,
        # and we hand it back so phase 2 renders against that exact one.

    if _prepare_lock.locked():
        raise web.HTTPConflict(text="Already preparing a clip")
    async with _prepare_lock:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=str(SRC_DIR), env=env,
        )
        # Timeout + tree-kill: without this, a cut whose ffmpeg ever stalls
        # would hold proc.communicate() — and therefore _prepare_lock — forever,
        # wedging EVERY future prepare into "Already preparing a clip" until the
        # whole app is restarted. The lock releases on the raised exception.
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=240)
        except asyncio.TimeoutError:
            await _kill_tree(proc)
            raise web.HTTPGatewayTimeout(
                text="Preparing the clip stalled and was stopped. Check the source "
                     "file and ranges, then try again.")

    out = stdout.decode(errors="replace")
    if proc.returncode != 0:
        raise web.HTTPInternalServerError(text=f"prepare failed:\n{out[-800:]}")

    clip_dir = raw_clip = movie = recap_music = None
    for line in out.splitlines():
        if "Clip dir:" in line:
            clip_dir = line.split("Clip dir:")[-1].strip()
        elif "Raw clip:" in line:
            raw_clip = line.split("Raw clip:")[-1].strip()
        elif "Recap music:" in line:
            recap_music = line.split("Recap music:")[-1].strip()
        elif "Movie:" in line:
            movie = line.split("Movie:")[-1].strip()
    if not clip_dir or not raw_clip:
        raise web.HTTPInternalServerError(text="prepare produced no clip")

    _prepared["clip_dir"] = clip_dir
    _prepared["raw_clip_url"] = _rel_output_url(raw_clip)
    _prepared["video"] = video
    _prepared["ranges"] = ranges
    return web.json_response({
        "ok": True,
        # Echoed back with /run so a render can be refused when the ranges were
        # edited after cutting (the reused cut would be the old footage).
        "prepared_ranges": ranges,
        # The client holds on to clip_dir and sends it back with /run — the
        # server-side _prepared global alone is lost on an app restart, which
        # used to make /run silently render a fresh dir (auto thumbnail
        # instead of the hand-picked one).
        "clip_dir": clip_dir,
        "raw_clip_url": _prepared["raw_clip_url"],
        "movie": movie or "",
        # Recap only: the track phase 1 actually snapped the cut to. The client
        # pins it so a "random" pick does not change between the two phases.
        "recap_music": Path(recap_music).name if recap_music else "",
    })


def _expected_steps(args: list[str]) -> int:
    """How many '[n/N]' steps the run we're about to spawn will print.

    Mirrors main.py's total_steps formula, read off the flags actually being
    passed, so the UI draws the right number of segments from the first paint
    instead of jumping when the real total arrives on the '[1/N]' line.
    """
    def _flag(name: str):
        return args[args.index(name) + 1] if name in args else None
    selection = _flag("--selection") or "manual"
    voice = _flag("--voice") or "none"
    is_recap = selection == "spine"
    narration_led = voice == "storytelling"
    auto_ranges = selection in ("spine", "script")
    return (5
            + (2 if is_recap else 0)                            # recap: analysis + animated outro
            - (1 if (is_recap and "--no-outro" in args) else 0)  # ...minus the outro when it is off
            + (2 if narration_led else 0)                       # write the script + speak it
            + (1 if (selection == "scene" and "--ranges" not in args) else 0)
            + (1 if ("--remove-silence" in args and not auto_ranges) else 0)
            + (2 if voice == "commentary" else 0)               # narration beats + voiceover
            + (0 if "--no-thumbnail" in args else
               (1 if (voice == "commentary" or "--prepend-thumbnail" in args or auto_ranges)
                else 0))                                        # thumbnail
            + (1 if "--prepend-thumbnail" in args else 0))      # bake thumbnail into video


@routes.post("/comparison-plan")
async def comparison_plan(request: web.Request) -> web.Response:
    """Generate and review one shared story before either expensive render."""
    if (run_state.proc is not None and run_state.proc.returncode is None) or request.app.get("comparison_busy"):
        raise web.HTTPConflict(text="Wait for the current job to finish")
    data = await request.json()
    video = Path(_clean_path(data.get("video")))
    srt = _find_srt_for_video(video)
    if not video.is_file() or not srt:
        raise web.HTTPBadRequest(text="Choose a movie with subtitles")
    try:
        start, end = float(data["span_start"]), float(data["span_end"])
        duration = float(data.get("target_seconds", 60))
    except (KeyError, ValueError, TypeError):
        raise web.HTTPBadRequest(text="Enable a source window and choose its bounds")
    err = await _ensure_ollama()
    if err:
        raise web.HTTPServiceUnavailable(text=err)
    folder = OUTPUT_DIR / ("comparison-" + uuid.uuid4().hex[:8])
    args = [sys.executable, str(EDITORIAL_SCRIPT), "--video", str(video), "--srt", str(srt),
            "--title", str(data.get("source_title") or video.stem),
            "--model", str(data.get("ollama_model") or "nemotron-3-super:cloud"),
            "--start", str(start), "--end", str(end), "--duration", str(duration), "--folder", str(folder),
            "--feedback", str(data.get("editorial_feedback") or "")[:3000]]
    request.app["comparison_busy"] = True
    jobs = request.app.setdefault("comparison_jobs", {})
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "running"}
    async def work():
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(*args, cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=660)
            if proc.returncode:
                raise ValueError(stderr.decode(errors="replace")[-2000:])
            plan = json.loads(stdout.decode())
            for beat in plan["beats"]:
                beat["previews"] = [_rel_output_url(p) for p in beat["previews"]]
            jobs[job_id] = {"status": "complete", "plan": plan}
        except Exception as exc:
            jobs[job_id] = {"status": "failed", "error": str(exc) or "Plan generation timed out"}
        finally:
            if proc is not None and proc.returncode is None:
                await _kill_tree(proc)
            request.app["comparison_busy"] = False
    request.app.setdefault("comparison_tasks", set()).add(asyncio.create_task(work()))
    return web.json_response({"job_id": job_id}, status=202)


@routes.get("/comparison-plan/{job_id}")
async def comparison_status(request: web.Request) -> web.Response:
    job = request.app.get("comparison_jobs", {}).get(request.match_info["job_id"])
    if job is None:
        raise web.HTTPNotFound(text="Plan job no longer exists; rebuild after an app restart")
    return web.json_response(job)


@routes.post("/run")
async def start_run(request: web.Request) -> web.Response:
    if request.app.get("comparison_busy"):
        raise web.HTTPConflict(text="An editorial plan is still being generated")
    if run_state.proc is not None and run_state.proc.returncode is None:
        raise web.HTTPConflict(text="A clip is already running")

    data = await request.json()

    video = _clean_path(data.get("video"))
    ranges = (data.get("ranges") or "").strip()
    mode = data.get("mode") or ("narrated" if data.get("narrate") else "classic")
    # The axes are authoritative; `mode` is only the preset label the UI showed.
    _PRESET_AXES = {
        "classic":      ("manual", "none"),
        "narrated":     ("manual", "commentary"),
        "recap":        ("spine",  "music"),
        "storytelling": ("script", "storytelling"),
    }
    _def_sel, _def_voice = _PRESET_AXES.get(mode, ("manual", "none"))
    selection = data.get("selection") or _def_sel
    voice = data.get("voice") or _def_voice
    if selection not in ("manual", "scene", "spine", "script"):
        raise web.HTTPBadRequest(text=f"Unknown selection: {selection}")
    if voice not in ("none", "commentary", "storytelling", "music"):
        raise web.HTTPBadRequest(text=f"Unknown voice: {voice}")
    if (voice == "storytelling") != (selection == "script"):
        raise web.HTTPBadRequest(text="Storytelling voice and script selection must be chosen together.")
    if (data.get("editorial_file") or "").strip() and voice not in ("commentary", "storytelling"):
        raise web.HTTPBadRequest(text="Approved editorial requires commentary or storytelling voice.")
    auto_ranges = selection in ("spine", "script", "scene")
    if not video or not Path(video).is_file():
        raise web.HTTPBadRequest(text="Video file not found")
    # Recap can auto-build its ranges from the whole subtitle file, so it may
    # legitimately arrive with none; every other mode needs at least one.
    if not ranges and not auto_ranges:
        raise web.HTTPBadRequest(text="At least one range is required")

    args = ["--video", video]
    if ranges:
        args += ["--ranges", ranges]

    if data.get("source_title"):
        args += ["--source-title", str(data["source_title"])]
    # Title picked (or typed) in the thumbnail editor — overrides the LLM's.
    if (data.get("title") or "").strip():
        args += ["--title", str(data["title"]).strip()]
    args += ["--whisper-model", data.get("whisper_model") or "medium"]
    args += ["--ollama-model", data.get("ollama_model") or "nemotron-3-super:cloud"]
    args += ["--language", data.get("language") or "en"]
    if (data.get("editorial_file") or "").strip():
        args += ["--editorial-file", _clean_path(data["editorial_file"])]
    if data.get("comparison_style"):
        if not data.get("visual_reviewed"):
            raise web.HTTPBadRequest(text="Review the character evidence and frame previews first")
        # Resolve only a plan produced by this app, never a client-supplied output path.
        editorial_path = Path(_clean_path(data.get("editorial_file"))).resolve()
        if OUTPUT_DIR.resolve() not in editorial_path.parents or editorial_path.name != "editorial.json":
            raise web.HTTPBadRequest(text="Comparison plan is outside Movie Clipper output")
        try:
            reviewed = json.loads(editorial_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise web.HTTPBadRequest(text="Comparison plan no longer exists")
        if reviewed.get("comparison_style") != data["comparison_style"]:
            raise web.HTTPBadRequest(text="Comparison style does not match its plan")
        args += ["--clip-dir", str(editorial_path.parent)]
        if data["comparison_style"] == "short":
            args += ["--byline-only"]
        else:
            args += ["--no-dialogue-captions"]
    if data.get("size"):
        args += ["--size", str(data["size"])]

    watermark = data.get("watermark") or "mv-edits"
    args += ["--watermark", watermark]
    args += ["--no-end-card" if data.get("end_card") is False else "--end-card"]
    if data.get("end_card_duration") is not None:
        try:
            end_card_duration = float(data["end_card_duration"])
        except (TypeError, ValueError):
            return web.json_response({"error": "End-card duration must be 1.5–2.5 seconds"}, status=400)
        if not 1.5 <= end_card_duration <= 2.5:
            return web.json_response({"error": "End-card duration must be 1.5–2.5 seconds"}, status=400)
        args += ["--end-card-duration", str(end_card_duration)]

    args += ["--selection", selection, "--voice", voice]
    if voice in ("commentary", "storytelling") and data.get("tts_voice"):
        args += ["--tts-voice", str(data["tts_voice"])]
    if voice in ("commentary", "storytelling") and data.get("narration_volume") is not None:
        args += ["--narration-volume", str(data["narration_volume"])]
    if voice in ("commentary", "storytelling") and data.get("audio_mix"):
        args += ["--audio-mix", str(data["audio_mix"])]
    if voice in ("commentary", "storytelling") and data.get("dialogue_volume") is not None:
        args += ["--dialogue-volume", str(data["dialogue_volume"])]
    if auto_ranges:
        # Recap montage: main.py needs the SRT (beats + captions). Music is
        # optional — with a track the cuts snap to its beats and the dialogue
        # ducks under it, without one the cuts stay where the LLM put them.
        # The outro face-scan runs inside main.py either way.
        if data.get("target_seconds") is not None:
            args += ["--target-seconds", str(data["target_seconds"])]
        if (data.get("recap_model") or "").strip():
            args += ["--recap-model", str(data["recap_model"]).strip()]
        if data.get("no_outro"):
            args += ["--no-outro"]
        if data.get("outro_seconds") is not None:
            args += ["--outro-seconds", str(data["outro_seconds"])]
        if data.get("auto_outro_shots") is not None:
            try:
                args += ["--auto-outro-shots", str(int(data["auto_outro_shots"]))]
            except (TypeError, ValueError):
                pass
        if (data.get("outro_shots") or "").strip():
            args += ["--outro-shots", str(data["outro_shots"]).strip()]
        # Movie SPAN: scope the auto picker. Ticked curated scenes (a multi-window
        # "a-b; c-d" string) win over the single-window slider.
        curated = (data.get("curated_region") or "").strip()
        if curated:
            args += ["--span", curated]
        else:
            try:
                ss, se = float(data.get("span_start")), float(data.get("span_end"))
                if se > ss + 1:
                    def _hms(t):
                        return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"
                    args += ["--span", f"{_hms(ss)}-{_hms(se)}"]
            except (TypeError, ValueError):
                pass

    # Cold-open hook + speech-gap boundary snap (both apply across selections).
    if data.get("hook"):
        args += ["--hook"]
        if data.get("hook_count"):
            try:
                args += ["--hook-count", str(int(data["hook_count"]))]
            except (TypeError, ValueError):
                pass
    if data.get("no_snap_speech"):
        args += ["--no-snap-speech"]

    # Two-phase applies to EVERY run now (narrated or classic): reuse the
    # prepared cut + hand-picked thumbnail, and bake the thumbnail into the video
    # so it can be chosen as the Short's thumbnail on YouTube mobile. Never fall
    # back silently: rendering without the prepared dir means a re-cut with an
    # auto-generated thumbnail — not what the user picked.
    if data.get("use_prepared"):
        # Only the client's own prepared_dir — NEVER fall back to the server-side
        # _prepared global. That global outlives the page and can point at a
        # completely different clip from an earlier session, which renders that
        # clip's footage under this clip's ranges.
        prep = _resolve_prepared_dir(data.get("prepared_dir"))
        if prep is None:
            raise web.HTTPBadRequest(
                text="The prepared clip is no longer available (app restarted?) — "
                     "click '① Cut clip & pick thumbnail' again.")
        # The cut is fixed at prepare time; the ranges below drive captions and
        # narration. If they've diverged, the video and the dialogue would come
        # from different clips — refuse rather than mismatch.
        prepared_ranges = _prepared_ranges_for(prep, data.get("prepared_ranges"))
        if prepared_ranges is not None and prepared_ranges != ranges:
            raise web.HTTPBadRequest(
                text=f"The ranges changed since you cut the clip "
                     f"(cut from '{prepared_ranges}', now '{ranges}'). "
                     f"Click '① Cut clip & pick thumbnail' again to re-cut and "
                     f"re-pick the thumbnail.")
        args += ["--clip-dir", str(prep), "--prepend-thumbnail"]
    if data.get("no_srt"):
        args += ["--no-srt"]
    elif data.get("srt") and Path(str(data["srt"])).is_file():
        args += ["--srt", str(data["srt"])]
    # Output shape: 9:16 vertical by default; the Square checkbox forces 1:1.
    # Narration needs the vertical band beneath the video, so a narrated edit is
    # always vertical regardless of the checkbox.
    aspect = (data.get("aspect") or "vertical").strip()
    layout = aspect if aspect in ("vertical", "square", "landscape") else "vertical"
    # No mode may override the shape any more — that coupling is exactly what
    # made the thumbnail unchangeable in narrated edits.
    args += ["--shape", layout]
    # Thumbnail is opt-out now, in every mode. Off means main.py composes none
    # and bakes none — the two-phase picker is skipped client-side to match.
    if data.get("want_thumbnail") is False:
        args += ["--no-thumbnail"]
    else:
        # Say so explicitly: classic edits compose none unless asked.
        args += ["--thumbnail"]
        thumb_w, thumb_h = {
            "vertical":  (1080, 1920),
            "square":    (1080, 1080),
            "landscape": (1920, 1080),
        }[layout]
        args += ["--thumbnail-size", f"{thumb_w}x{thumb_h}"]
    if data.get("remove_silence"):
        args += ["--remove-silence"]
    if data.get("no_loudnorm"):
        args += ["--no-loudnorm"]

    # Caption style options
    if data.get("caption_font"):
        args += ["--caption-font", str(data["caption_font"])]
    if data.get("caption_animation"):
        args += ["--caption-animation", str(data["caption_animation"])]
    if data.get("caption_color"):
        args += ["--caption-color", str(data["caption_color"])]
    if data.get("caption_shuffle") is False:
        args += ["--no-caption-shuffle"]
    if data.get("caption_font_shuffle"):
        args += ["--caption-font-shuffle"]
    if data.get("caption_italic"):
        args += ["--caption-italic"]
    if data.get("mirror"):
        args += ["--mirror"]

    bg_mode = data.get("bg_mode") or "none"
    if bg_mode == "none":
        args += ["--no-bg-music"]
    elif bg_mode == "file" and data.get("bg_music"):
        bg_path = AUDIO_DIR / str(data["bg_music"])
        if bg_path.is_file():
            args += ["--bg-music", str(bg_path)]
    if data.get("bg_volume") is not None:
        args += ["--bg-volume", str(data["bg_volume"])]
    # Skip a lead-in on the background track (random or specific file); ignored
    # when bg music is off.
    if bg_mode != "none" and data.get("bg_skip") is not None:
        try:
            if float(data["bg_skip"]) > 0:
                args += ["--bg-skip", str(data["bg_skip"])]
        except (TypeError, ValueError):
            pass

    # Edit-style description credits. Recap defaults to it in main.py, so an
    # unchecked box must explicitly opt out (--no-edit-credits) to override that.
    if data.get("edit_credits"):
        args += ["--edit-credits"]
        if (data.get("music_credit") or "").strip():
            args += ["--music-credit", str(data["music_credit"]).strip()]
        if (data.get("editing_program") or "").strip():
            args += ["--editing-program", str(data["editing_program"]).strip()]
        if data.get("no_disclaimer"):
            args += ["--no-disclaimer"]
    else:
        args += ["--no-edit-credits"]

    if not data.get("editorial_file") or selection in ("scene", "spine") or data.get("hook"):
        ollama_err = await _ensure_ollama()
        if ollama_err:
            raise web.HTTPServiceUnavailable(text=ollama_err)

    run_state.reset()
    run_state.total_steps = _expected_steps(args)
    asyncio.create_task(_stream_run(args))

    return web.json_response({"ok": True, "total_steps": run_state.total_steps})


@routes.get("/ws")
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    run_state.subscribers.add(ws)

    for line in run_state.lines:
        await ws.send_str(json.dumps({
            "type": "line", "text": line,
            "step": run_state.step, "total": run_state.total_steps,
            "step_text": run_state.step_text,
        }))
    if run_state.done:
        await ws.send_str(json.dumps({"type": "done", "ok": run_state.ok, "result": run_state.result}))

    async for msg in ws:
        if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
            break

    run_state.subscribers.discard(ws)
    return ws


@routes.get("/output/{path:.*}")
async def serve_output(request: web.Request) -> web.StreamResponse:
    rel = request.match_info["path"]
    full = (OUTPUT_DIR / rel).resolve()
    out_root = OUTPUT_DIR.resolve()
    if full != out_root and out_root not in full.parents:
        raise web.HTTPForbidden()
    if not full.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(full)


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and its whole child tree (the spawned ffmpeg). On
    Windows terminate() only signals the direct child, so taskkill /T is needed
    to reap the ffmpeg grandchild too."""
    if proc.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
        await asyncio.wait_for(proc.wait(), timeout=5)
    except Exception:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def _terminate_run(app: web.Application) -> None:
    """On shutdown, kill any in-flight render and its whole child tree so a
    stopped/restarted server (e.g. Agent Hub cycling the app) never leaves
    orphaned ffmpeg/edge-tts processes behind."""
    proc = run_state.proc
    if proc is None or proc.returncode is not None:
        return
    await _kill_tree(proc)


async def _mute_reset_noise(app: web.Application) -> None:
    # Windows' ProactorEventLoop logs a ConnectionResetError traceback whenever
    # a client (phone backgrounding Safari, losing signal, etc.) drops the
    # socket instead of closing it cleanly. It's not an actual failure — just
    # noise. Installed on_startup so it binds the *actual* serving loop (calling
    # asyncio.get_event_loop() at build time grabbed a throwaway loop and raised
    # a DeprecationWarning on 3.12+).
    try:
        loop = asyncio.get_running_loop()
        _prev = loop.get_exception_handler()

        def _exc_handler(lp: asyncio.AbstractEventLoop, ctx: dict) -> None:
            if isinstance(ctx.get("exception"), ConnectionResetError):
                return
            if _prev:
                _prev(lp, ctx)
            else:
                lp.default_exception_handler(ctx)

        loop.set_exception_handler(_exc_handler)
    except Exception:
        pass


async def _sigint_wakeup(app: web.Application) -> None:
    """Keep Ctrl+C responsive on Windows.

    The ProactorEventLoop blocks in GetQueuedCompletionStatus and only processes
    SIGINT when it wakes for socket activity — so on an idle server Ctrl+C seems
    to do nothing and you're forced to kill the process from Task Manager. A tiny
    periodic sleep keeps the loop waking so the interrupt is handled at once.
    """
    async def _tick() -> None:
        try:
            while True:
                await asyncio.sleep(0.4)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(_tick())

    async def _stop(_app: web.Application) -> None:
        task.cancel()

    app.on_cleanup.append(_stop)


def create_app() -> web.Application:
    # client_max_size: /make-thumbnail receives a base64 canvas capture of the
    # displayed frame — a 4K source frame can exceed aiohttp's 1MB default.
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes(routes)
    app.on_startup.append(_mute_reset_noise)
    if sys.platform == "win32":
        app.on_startup.append(_sigint_wakeup)
    app.on_cleanup.append(_terminate_run)
    return app


def main() -> None:
    app = create_app()
    # HOST is the *bind* address (0.0.0.0 = all interfaces). A browser can't
    # connect to 0.0.0.0 — so advertise a URL that actually resolves.
    open_host = "localhost" if HOST in ("0.0.0.0", "::", "") else HOST
    print(f"Movie Clipper UI -> http://{open_host}:{PORT}")
    if HOST == "0.0.0.0":
        print(f"  (open this in your browser: http://localhost:{PORT}  or  http://127.0.0.1:{PORT})")
    print("  Press Ctrl+C in this window to stop the server.")
    web.run_app(app, host=HOST, port=PORT, print=None)


# ── HTML ────────────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Movie Clipper</title>
<link rel="icon" type="image/svg+xml" href="__BASE_PATH__/favicon.svg">
<link rel="apple-touch-icon" href="__BASE_PATH__/favicon.png">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0e14;--panel:#12161f;--border:#232838;--border-bright:#3a4258;
  --accent:#ff7a30;--text:#e7e9ee;--text-muted:#8a91a3;--text-faint:#565d70;
  --ok:#3ddc84;--err:#ff5c5c;
}
html,body{min-height:100%;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif}
body{padding:20px 14px 60px;max-width:640px;margin:0 auto}
h1{font-size:18px;font-weight:600;letter-spacing:.3px;margin-bottom:2px}
.sub{color:var(--text-muted);font-size:12.5px;margin-bottom:20px}
fieldset{border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:14px}
legend{font-size:11px;letter-spacing:.6px;color:var(--text-muted);padding:0 6px;text-transform:uppercase}
label{display:block;font-size:12px;color:var(--text-muted);margin:10px 0 4px}
label:first-child{margin-top:0}
input[type=text],input[type=number],select{
  width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;
  color:var(--text);padding:9px 10px;font-size:13.5px;outline:none;transition:border-color .15s;
}
input:focus,select:focus{border-color:var(--accent)}
input[type=range]{width:100%;accent-color:var(--accent)}
.row{display:flex;gap:8px}
.row>*{flex:1}
.range-row{display:flex;gap:8px;align-items:center;margin-bottom:6px}
.range-row input{flex:1}
.range-row button{flex-shrink:0}
.mini-btn{background:transparent;border:1px solid var(--border-bright);color:var(--text-muted);
  border-radius:5px;padding:8px 10px;font-size:12px;cursor:pointer;transition:all .15s}
.mini-btn:hover{border-color:var(--accent);color:var(--accent)}
#add-range{font-size:12px;margin-top:2px}
details{margin-top:4px}
summary{font-size:12px;color:var(--text-muted);cursor:pointer;padding:4px 0}
#volume-val{font-size:11px;color:var(--text-faint);float:right}
#submit{width:100%;background:var(--accent);color:#1a0e05;border:none;border-radius:8px;
  padding:13px;font-size:14px;font-weight:700;cursor:pointer;margin-top:4px;transition:opacity .15s}
#submit:hover:not(:disabled){opacity:.88}
#submit:disabled{opacity:.4;cursor:not-allowed}
#progress{display:none;margin-top:18px}
#step-label{font-size:12.5px;font-weight:600;color:var(--accent);margin-bottom:8px;min-height:16px}
#step-label.done{color:var(--ok)}
#step-label.failed{color:var(--err)}
.steps{display:flex;gap:4px;margin-bottom:12px}
.step{flex:1;height:5px;border-radius:3px;background:var(--border)}
.step.active{background:var(--accent)}
.step.past{background:var(--ok)}
#log{background:#000;border:1px solid var(--border);border-radius:6px;padding:10px;
  height:220px;overflow-y:auto;font-family:ui-monospace,Consolas,monospace;font-size:11.5px;
  color:var(--text-muted);white-space:pre-wrap;line-height:1.5}
#result{display:none;margin-top:16px;border:1px solid var(--ok);border-radius:8px;padding:14px}
#result video{width:100%;border-radius:6px;background:#000;margin-bottom:10px}
#result h2{font-size:14px;margin-bottom:6px}
#result a{color:var(--accent);font-size:12.5px;display:inline-block;margin-right:14px}
#error{display:none;margin-top:14px;border:1px solid var(--err);color:var(--err);
  border-radius:6px;padding:10px;font-size:12.5px}
#srt-status{font-size:11.5px;margin-top:5px;color:var(--text-faint)}
#srt-status.found{color:var(--ok)}
#suggest-btn{display:block;font-size:12px;margin-top:8px}
.suggestion{border:1px solid var(--border);border-radius:7px;padding:10px 12px;margin-top:8px;
  cursor:pointer;transition:border-color .15s}
.suggestion:hover{border-color:var(--accent)}
.suggestion.added{border-color:var(--ok)}
.suggestion .s-head{display:flex;justify-content:space-between;gap:8px;font-size:13px}
.suggestion .s-range{color:var(--accent);font-family:ui-monospace,Consolas,monospace;font-size:12px;white-space:nowrap}
.suggestion .s-reason{color:var(--text-muted);font-size:11.5px;margin-top:4px;line-height:1.45}
#result img{width:100%;border-radius:6px;margin-bottom:10px}
#thumb-editor{margin-top:16px;border-top:1px solid var(--border);padding-top:12px}
#thumb-render{width:100%;background:var(--accent);color:#1a0e05;border:none;border-radius:8px;
  padding:12px;font-size:14px;font-weight:700;cursor:pointer;margin-top:12px}
#thumb-render:disabled{opacity:.5;cursor:not-allowed}
.thumb-head{font-size:13px;font-weight:600;margin-bottom:8px}
.thumb-hint{font-size:11.5px;color:var(--text-muted);margin-bottom:8px;line-height:1.45}
#thumb-video-wrap{position:relative;line-height:0;margin-bottom:8px}
#thumb-video{width:100%;border-radius:6px;background:#000;display:block}
#thumb-cropbox{position:absolute;top:0;height:100%;box-sizing:border-box;
  border:2px solid var(--accent);background:rgba(255,122,48,0.10);
  box-shadow:0 0 0 3000px rgba(0,0,0,0.42);pointer-events:none;display:none}
#result-thumb{max-width:270px;display:block;margin:0 auto 10px;border:1px solid var(--border-bright)}
.thumb-status{font-size:11.5px;color:var(--text-faint);margin-left:10px}
.thumb-status.ok{color:var(--ok)}

/* ── Visual layer ────────────────────────────────────────────────────────────
   Everything below refines the base rules above: a warmer, deeper ground, a
   real header, format cards instead of a bare <select>, switch-style toggles,
   and section chrome that makes the per-mode show/hide legible at a glance. */
:root{
  --accent2:#ffb648;          /* amber end of the accent gradient */
  --violet:#7c6cff;
  --panel-2:#161b26;
  --glow:0 0 0 1px rgba(255,122,48,.35), 0 8px 28px -10px rgba(255,122,48,.55);
}
html,body{
  background:
    radial-gradient(900px 420px at 12% -8%, rgba(255,122,48,.13), transparent 62%),
    radial-gradient(760px 400px at 92% 4%, rgba(124,108,255,.11), transparent 60%),
    var(--bg);
  background-attachment:fixed;
}
body{max-width:680px;padding:26px 15px 72px}

/* Header */
.hero{display:flex;align-items:center;gap:13px;margin-bottom:22px}
.hero-mark{
  width:44px;height:44px;flex-shrink:0;border-radius:13px;display:grid;place-items:center;
  font-size:22px;background:linear-gradient(145deg,var(--accent),var(--accent2));
  box-shadow:0 8px 24px -8px rgba(255,122,48,.75);
}
h1{
  font-size:21px;font-weight:750;letter-spacing:-.2px;line-height:1.1;
  background:linear-gradient(92deg,var(--text) 30%,var(--accent2));
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
}
.sub{margin-bottom:0;font-size:12.5px}

/* Sections */
fieldset{
  background:linear-gradient(180deg,var(--panel-2),var(--panel));
  border-color:var(--border);border-radius:13px;padding:16px 15px;margin-bottom:13px;
  box-shadow:0 1px 0 rgba(255,255,255,.03) inset;
}
fieldset[data-mode-only]{border-left:2px solid var(--accent)}
legend{
  font-size:10.5px;font-weight:700;letter-spacing:1.1px;color:var(--text-muted);
  background:var(--bg);border:1px solid var(--border);border-radius:20px;padding:3px 10px;
}
label{font-size:11.5px;letter-spacing:.2px}
input[type=text],input[type=number],select{background:rgba(0,0,0,.32);border-radius:8px;padding:10px 11px}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(255,122,48,.13)}
select{appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--text-muted) 50%),
  linear-gradient(135deg,var(--text-muted) 50%,transparent 50%);
  background-position:calc(100% - 17px) 50%,calc(100% - 12px) 50%;
  background-size:5px 5px,5px 5px;background-repeat:no-repeat;padding-right:32px}

/* Format cards */
.fmt-cards{display:flex;flex-direction:column;gap:8px}
.fmt{
  display:flex;gap:12px;align-items:flex-start;text-align:left;width:100%;cursor:pointer;
  background:rgba(0,0,0,.3);border:1px solid var(--border);border-radius:11px;
  padding:12px 13px;color:var(--text);font:inherit;transition:border-color .16s,background .16s,transform .1s;
}
.fmt:hover{border-color:var(--border-bright)}
.fmt:active{transform:scale(.994)}
.fmt[aria-pressed="true"]{
  /* Shorthand, not border-color: the base .fmt rule sets `border` as a shorthand
     and a longhand override here loses to it in some engines. */
  border:1px solid var(--accent);
  background:linear-gradient(180deg,rgba(255,122,48,.13),rgba(255,122,48,.05));
  box-shadow:var(--glow);
}
.fmt-ico{font-size:19px;line-height:1.25;flex-shrink:0}
.fmt-body{display:block;min-width:0}
.fmt-name{display:block;font-size:14px;font-weight:700;letter-spacing:-.1px;margin-bottom:3px}
.fmt-desc{display:block;font-size:11.5px;color:var(--text-muted);line-height:1.5;margin-bottom:7px}

/* Pills */
.pill{
  display:inline-block;font-size:10px;font-weight:600;letter-spacing:.3px;
  border:1px solid var(--border-bright);color:var(--text-muted);
  border-radius:20px;padding:2.5px 8px;margin-right:5px;white-space:nowrap;
}
.pill.good{border-color:rgba(61,220,132,.45);color:var(--ok);background:rgba(61,220,132,.08)}
.pill.warn{border-color:rgba(255,182,72,.45);color:var(--accent2);background:rgba(255,182,72,.08)}
.pill.audio{border-color:rgba(255,122,48,.4);color:var(--accent);background:rgba(255,122,48,.07)}

/* Switch-style checkboxes */
.switch{display:flex;align-items:center;gap:11px;cursor:pointer;margin:12px 0 0;
  font-size:12.5px;color:var(--text);line-height:1.45}
.switch:first-child{margin-top:0}
.switch input{appearance:none;flex-shrink:0;width:38px;height:22px;border-radius:20px;
  background:rgba(255,255,255,.09);border:1px solid var(--border-bright);
  position:relative;cursor:pointer;transition:background .18s,border-color .18s;margin:0}
.switch input::after{content:"";position:absolute;top:2px;left:2px;width:16px;height:16px;
  border-radius:50%;background:var(--text-muted);transition:transform .18s,background .18s}
.switch input:checked{background:linear-gradient(135deg,var(--accent),var(--accent2));border-color:transparent}
.switch input:checked::after{transform:translateX(16px);background:#1a0e05}
.switch .sw-text{display:block}
.switch .sw-sub{display:block;font-size:11px;color:var(--text-faint);margin-top:2px;line-height:1.45}

/* Notes + hints */
.note{font-size:11.5px;color:var(--text-faint);line-height:1.55;margin-top:8px}
.note.lock{color:var(--accent2)}
.note strong{color:var(--text-muted);font-weight:600}
.readout{float:right;font-size:11px;color:var(--accent);font-weight:600;
  font-family:ui-monospace,Consolas,monospace}
.readout.over{color:var(--err)}
.divider{height:1px;background:var(--border);margin:14px 0 2px}

/* Segmented button group (replaces small style dropdowns) */
.seg-group{display:flex;gap:6px;flex-wrap:wrap;margin-top:2px}
.seg{flex:1;min-width:74px;padding:9px 8px;border-radius:8px;border:1px solid var(--border-bright);
  background:rgba(255,255,255,.04);color:var(--text-muted);font-weight:600;font-size:12.5px;cursor:pointer;
  transition:background .15s,border-color .15s,color .15s}
.seg:hover{border-color:var(--accent2)}
.seg.active{background:rgba(255,122,48,.14);border-color:var(--accent);color:var(--accent)}

/* Buttons */
.mini-btn{border-radius:7px;font-weight:600}
.mini-btn.primary{border-color:rgba(255,122,48,.5);color:var(--accent);background:rgba(255,122,48,.09)}
.mini-btn.primary:hover{background:rgba(255,122,48,.16)}
#ai-btn{display:block;width:100%;margin-top:10px;padding:11px;font-size:12.5px}
#submit{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:11px;padding:15px;font-size:14.5px;letter-spacing:.2px;
  box-shadow:0 10px 30px -12px rgba(255,122,48,.85);margin-top:8px;
}
#submit-hint{text-align:center;font-size:11.5px;color:var(--text-faint);margin-top:9px;line-height:1.5}
#thumb-render{background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:10px}

/* Progress + result */
.step{height:6px;border-radius:4px}
.step.active{background:linear-gradient(90deg,var(--accent),var(--accent2))}
#result{border-radius:13px;box-shadow:0 12px 40px -20px rgba(61,220,132,.6)}
#log{border-radius:8px;background:#04060a}
#error{border-radius:9px;background:rgba(255,92,92,.07)}
.suggestion{border-radius:9px;background:rgba(0,0,0,.28)}
.suggestion:hover{background:rgba(255,122,48,.06)}
#srt-status{font-size:12px;margin-top:8px;padding:9px 11px;border-radius:8px;
  border:1px solid var(--border);background:rgba(0,0,0,.28)}
#srt-status.found{color:var(--ok);border-color:rgba(61,220,132,.35);background:rgba(61,220,132,.07)}
#srt-status.missing{color:var(--accent2);border-color:rgba(255,182,72,.32);background:rgba(255,182,72,.06)}
@media (max-width:420px){
  .fmt-desc{font-size:11px}
  body{padding:20px 12px 64px}
}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-mark">🎬</div>
  <div>
    <h1>Movie Clipper</h1>
    <div class="sub">Cut, caption, and upload-ready in one run</div>
  </div>
</header>

<form id="form">
  <fieldset>
    <legend>① WHAT ARE YOU MAKING</legend>
    <div class="fmt-cards" id="fmt-cards">
      <button type="button" class="fmt" data-value="narrated" aria-pressed="true">
        <span class="fmt-ico">🎙️</span>
        <span class="fmt-body">
          <span class="fmt-name">Narrated Short</span>
          <span class="fmt-desc">One scene with a voiceover talking over it, text band beneath. The clip you pick sets the length.</span>
          <span class="pill audio">Voiceover over full audio</span><span class="pill good">Transformative</span><span class="pill">Your ranges</span><span class="pill">Vertical</span>
        </span>
      </button>
      <button type="button" class="fmt" data-value="storytelling" aria-pressed="false">
        <span class="fmt-ico">📖</span>
        <span class="fmt-body">
          <span class="fmt-name">Storytelling Recap</span>
          <span class="fmt-desc">The narrator retells the whole plot over clips, as a lower third. <strong>The script sets the length</strong> — each clip runs exactly as long as its line.</span>
          <span class="pill audio">Narrator leads · movie muted</span><span class="pill good">Narrator-led storytelling</span><span class="pill">Script picks clips</span><span class="pill">Landscape</span>
        </span>
      </button>
      <button type="button" class="fmt" data-value="recap" aria-pressed="false">
        <span class="fmt-ico">⚡</span>
        <span class="fmt-body">
          <span class="fmt-name">Recap Montage</span>
          <span class="fmt-desc">Short plot beats cut back to back, carried by music with the dialogue ducked under it. No narrator.</span>
          <span class="pill audio">Music bed · dialogue ducked</span><span class="pill good">Transformative</span><span class="pill">Beats picked for you</span>
        </span>
      </button>
      <button type="button" class="fmt" data-value="classic" aria-pressed="false">
        <span class="fmt-ico">✂️</span>
        <span class="fmt-body">
          <span class="fmt-name">Classic</span>
          <span class="fmt-desc">A straight cut with karaoke captions. Nothing added on top.</span>
          <span class="pill audio">Original audio kept</span><span class="pill warn">Fair-use only</span><span class="pill">Your ranges</span>
        </span>
      </button>
    </div>
    <select id="format" style="display:none" aria-hidden="true" tabindex="-1">
      <option value="narrated" selected>Narrated Short</option>
      <option value="storytelling">Storytelling Recap</option>
      <option value="recap">Recap Montage</option>
      <option value="classic">Classic</option>
      <option value="custom">Custom</option>
    </select>

    <details id="axes-details">
      <summary>Mix your own — these presets are just three settings</summary>
      <div class="note" style="margin-bottom:4px">
        Every combination is buildable. A preset only fills these in for you; change one and
        you get a custom edit.
      </div>
      <label for="sel_selection">Who picks the clips</label>
      <select id="sel_selection">
        <option value="manual">You do — the ranges below</option>
        <option value="scene">Pick the best single scene for me</option>
        <option value="spine">Pick a chain of plot beats for me</option>
        <option value="script">The narration script picks them</option>
      </select>
      <label for="sel_voice">What carries the edit</label>
      <select id="sel_voice">
        <option value="none">Just the film's own dialogue</option>
        <option value="commentary">Voiceover under full dialogue — commentary on one moment</option>
        <option value="storytelling">Voiceover in front, dialogue ducked — retells the plot</option>
        <option value="music">A music bed with the dialogue ducked under it</option>
      </select>
      <div id="axis-note" class="note"></div>
    </details>
  </fieldset>

  <fieldset>
    <legend>② SOURCE</legend>
    <label for="video">Movie file path</label>
    <div class="range-row">
      <input type="text" id="video" placeholder="N:\\Movies\\film.mkv" required>
      <button type="button" id="check-srt" class="mini-btn">Re-check</button>
    </div>
    <div id="srt-status">Paste a movie path — subtitles are looked up automatically.</div>

    <label class="switch" style="margin-top:10px">
      <input type="checkbox" id="span_enable">
      <span class="sw-text">Summarize only part of the movie
        <span class="sw-sub">The clip-picker reads &amp; picks from just this stretch — render sections separately. Bounds set from the movie's length after the check above.</span>
      </span>
    </label>
    <div id="span-wrap" style="display:none">
      <label>From <span id="span-from-val" class="readout">0:00</span></label>
      <input type="range" id="span_start" min="0" max="6000" step="1" value="0">
      <label>To <span id="span-to-val" class="readout">end</span></label>
      <input type="range" id="span_end" min="0" max="6000" step="1" value="6000">
    </div>

    <label for="source_title">Source title override (optional)</label>
    <input type="text" id="source_title" placeholder="Auto-parsed from filename if left blank">
  </fieldset>

  <fieldset id="subs-fieldset">
    <legend>③ SUBTITLES &amp; SCENE AI</legend>
    <label class="switch" id="use-srt-wrap" style="display:none">
      <input type="checkbox" id="use_srt" checked>
      <span class="sw-text">Use this subtitle file
        <span class="sw-sub">Drives the burned captions, the scene picks, and the narration script — all locally, nothing uploaded.</span>
      </span>
    </label>
    <div id="subs-none" class="note">
      No subtitle file yet. Captions can still be transcribed with Whisper, but the
      scene AI below needs an <strong>.srt</strong> next to the movie.
    </div>
    <div id="suggest-wrap" style="display:none">
      <div class="divider"></div>
      <label for="suggest_model">Model that reads the subtitle file</label>
      <select id="suggest_model">
        <option value="gemma4:e4b" selected>gemma4:e4b — local, nothing uploaded</option>
        <option value="gemma4:e2b">gemma4:e2b — local, smaller and faster</option>
        <option value="__cloud__">Cloud — the Ollama model under Generation (uploads the subtitles)</option>
      </select>
      <button type="button" id="ai-btn" class="mini-btn primary">✨ Suggest scenes from subtitles</button>
      <div id="ai-hint" class="note">The whole subtitle file goes to the model; it comes back with clip ranges you can click to add.</div>
      <div id="ai-status" class="note"></div>
    </div>
    <div id="suggestions"></div>
  </fieldset>

  <fieldset>
    <legend>④ CLIPS &amp; LENGTH</legend>
    <div id="ranges-wrap">
      <label>Clip range(s) <span id="ranges-total" class="readout"></span></label>
      <div id="ranges"></div>
      <button type="button" id="add-range" class="mini-btn">+ Add range</button>
      <div id="ranges-note" class="note"></div>
      <div class="divider"></div>
    </div>
    <label for="target_len">Target length <span id="target-len-val" class="readout">120s</span></label>
    <input type="range" id="target_len" min="30" max="180" step="10" value="120">
    <div id="length-hint" class="note"></div>

    <div class="divider"></div>
    <label class="switch">
      <input type="checkbox" id="hook">
      <span class="sw-text">Cold-open hook
        <span class="sw-sub">Open with the 1-3 punchiest lines as a teaser, before the main content.</span>
      </span>
    </label>
    <label class="switch">
      <input type="checkbox" id="snap_speech" checked>
      <span class="sw-text">Snap cuts to speech gaps
        <span class="sw-sub">Verify clip boundaries against the audio so lines don't cut mid-sentence (adds a short pass).</span>
      </span>
    </label>
  </fieldset>

  <fieldset id="narrate-fieldset" data-mode-only>
    <legend>NARRATION</legend>
    <input type="checkbox" id="narrate" checked style="display:none">
    <label for="tts_voice">Narrator voice</label>
    <select id="tts_voice">
      <option value="en-US-JennyNeural" selected>Jenny (US, female)</option>
      <option value="en-US-ChristopherNeural">Christopher (US, deep)</option>
      <option value="en-US-GuyNeural">Guy (US)</option>
      <option value="en-GB-RyanNeural">Ryan (British)</option>
      <option value="en-GB-SoniaNeural">Sonia (British)</option>
      <option value="en-AU-WilliamNeural">William (Australian)</option>
    </select>
    <div id="narrate-opts">
      <div id="narrate-kind" class="note"></div>
      <div id="audio-mix-wrap">
        <label for="audio_mix">Commentary audio mix</label>
        <select id="audio_mix"><option value="narrator-led" selected>Narrator leads: film ducks</option><option value="original">Original: film leads (narration sits ~25 dB under it)</option><option value="narration-only">Narration only: source soundtrack muted</option></select>
        <div class="note">Review separately: original contribution, footage/audio rights, and ad suitability. This control does not clear copyright claims.</div>
      </div>
      <div id="narrate-level-wrap">
        <label for="narration_volume"><span id="nvol-label">Voiceover volume</span> <span id="nvolume-val" class="readout">0.04</span></label>
        <input type="range" id="narration_volume" min="0" max="1" step="0.01" value="0.04">
      </div>
      <div id="dialogue-level-wrap" style="display:none">
        <label for="dialogue_volume">Dialogue under the narrator <span id="dvolume-val" class="readout">0.22</span></label>
        <input type="range" id="dialogue_volume" min="0" max="1" step="0.01" value="0.22">
        <div class="note">How loud the film sits beneath the voice. It ducks further
          automatically whenever the narrator is speaking.</div>
      </div>
    </div>
  </fieldset>

  <fieldset id="script-fieldset" data-mode-only style="display:none">
    <legend>NARRATION SCRIPT</legend>
    <div class="note">
      The model reads the whole subtitle file and writes a connected retelling of the plot.
      That script is then spoken, and <strong>each clip is cut to last exactly as long as its
      line takes to read</strong> — so the target length above is a length of <em>narration</em>,
      not of footage, and you don't pick ranges at all.
      <br>Because the clips don't exist until the script has been spoken, this is the one
      edit that can't stop for a hand-picked thumbnail.
    </div>
  </fieldset>

  <fieldset id="recap-fieldset" data-mode-only style="display:none">
    <legend>RECAP MONTAGE</legend>
    <div class="note">
      Needs a subtitle file — it drives both the beat picks and the captions.
      <strong>Leave the ranges empty</strong> to recap the whole movie, add one long
      range to scope it to a stretch, or hit the AI button above to preview and tweak the beats.
      <br>Pick a music track under <em>Background music</em> and the cuts snap to its beats
      with the dialogue ducked underneath. Leave music off and the cuts land exactly
      where the model put them, carried by the movie's own audio.
    </div>
    <div class="divider"></div>
    <label class="switch">
      <input type="checkbox" id="recap_outro">
      <span class="sw-text">Iconic outro montage <span style="color:var(--text-faint)">(beta)</span>
        <span class="sw-sub">Rapid face-shot stills on the final seconds, beat-synced when there is music. Off by default.</span>
      </span>
    </label>
    <div id="recap-outro-opts" style="display:none">
      <label for="outro_seconds">Outro length <span id="outro-sec-val" class="readout">5s</span></label>
      <input type="range" id="outro_seconds" min="2" max="12" step="1" value="5">
      <label for="auto_outro_shots">Auto-find this many iconic face shots (0 = off, scans at render)</label>
      <input type="number" id="auto_outro_shots" min="0" max="30" step="1" value="12">
      <label for="outro_shots_manual">Extra iconic timestamps (optional, comma-separated)</label>
      <input type="text" id="outro_shots_manual" placeholder="1:02:10, 1:15:03">
    </div>
  </fieldset>

  <fieldset>
    <legend>⑤ OUTPUT &amp; THUMBNAIL</legend>
    <label for="aspect">Output shape</label>
    <select id="aspect">
      <option value="vertical" selected>9:16 vertical — Shorts / TikTok</option>
      <option value="square">1:1 square</option>
      <option value="landscape">16:9 landscape — long-form YouTube</option>
    </select>
    <div id="aspect-note" class="note"></div>
    <div class="divider"></div>
    <label class="switch">
      <input type="checkbox" id="want_thumbnail">
      <span class="sw-text">Pick the thumbnail by hand
        <span class="sw-sub" id="thumb-shape-note"></span>
      </span>
    </label>
  </fieldset>

  <fieldset>
    <legend>⑥ CAPTIONS</legend>
    <label for="caption_font">Font</label>
    <select id="caption_font">
      <option value="Anton">Anton ★ — bundled, viral-shorts caps</option>
      <option value="Bebas Neue">Bebas Neue ★ — bundled, tall condensed</option>
      <option value="Impact">Impact</option>
      <option value="Arial Black" selected>Arial Black</option>
      <option value="Verdana">Verdana</option>
      <option value="Tahoma">Tahoma</option>
      <option value="Trebuchet MS">Trebuchet MS</option>
      <option value="Georgia">Georgia</option>
      <option value="Comic Sans MS">Comic Sans MS</option>
    </select>
    <label class="switch">
      <input type="checkbox" id="caption_font_shuffle">
      <span class="sw-text">Shuffle fonts
        <span class="sw-sub">Cycle a different typeface each sentence, starting from the font above. Works off the subtitle file or a Whisper transcript.</span>
      </span>
    </label>
    <label class="switch">
      <input type="checkbox" id="caption_italic">
      <span class="sw-text">Italic captions
        <span class="sw-sub">Render the burned captions slanted.</span>
      </span>
    </label>
    <label>Effect</label>
    <div class="seg-group" id="caption_animation_btns">
      <button type="button" class="seg active" data-value="karaoke">Karaoke</button>
      <button type="button" class="seg" data-value="fade">Fade</button>
      <button type="button" class="seg" data-value="pop">Pop</button>
    </div>
    <input type="hidden" id="caption_animation" value="karaoke">
    <label class="switch">
      <input type="checkbox" id="caption_shuffle" checked>
      <span class="sw-text">Shuffle colours
        <span class="sw-sub">Cycle a different colour each sentence.</span>
      </span>
    </label>
    <div id="caption_color_wrap" style="display:none">
      <label for="caption_color">Fixed colour</label>
      <select id="caption_color">
        <option value="white" selected>White</option>
        <option value="yellow">Yellow</option>
        <option value="green">Green</option>
        <option value="cyan">Cyan</option>
        <option value="pink">Pink</option>
        <option value="orange">Orange</option>
      </select>
    </div>
  </fieldset>

  <fieldset>
    <legend>CHANNEL END CARD</legend>
    <label><input type="checkbox" id="end_card" checked> End with channel identity and Subscribe</label>
    <label for="end_card_duration">End-card seconds</label>
    <input type="number" id="end_card_duration" min="1.5" max="2.5" step="0.1" value="2">
    <div class="note">Adds this time after the edit. Branding and wording come from brand.json. Turn off for a fixed-runtime comparison.</div>
  </fieldset>

  <fieldset>
    <legend>⑦ BACKGROUND MUSIC</legend>
    <label for="bg_mode">Mode</label>
    <select id="bg_mode">
      <option value="none" selected>None</option>
      <option value="random">Random from audio/</option>
      <option value="file">Specific file</option>
    </select>
    <div id="music-note" class="note"></div>
    <div id="bg-file-wrap" style="display:none">
      <label for="bg_music">File</label>
      <select id="bg_music"></select>
    </div>
    <div id="bg-audio-opts" style="display:none">
      <label for="bg_volume">Volume <span id="volume-val" class="readout">0.10</span></label>
      <input type="range" id="bg_volume" min="0" max="1" step="0.01" value="0.10">
      <label for="bg_skip">Skip first N seconds of the track (cut a slow intro)</label>
      <input type="number" id="bg_skip" min="0" step="0.5" value="0" placeholder="0">
    </div>
  </fieldset>

  <fieldset>
    <legend>⑧ GENERATION</legend>
    <div class="row">
      <div id="whisper-wrap">
        <label for="whisper_model">Whisper model (no subtitles)</label>
        <select id="whisper_model">
          <option value="tiny">tiny</option>
          <option value="base">base</option>
          <option value="small">small</option>
          <option value="medium" selected>medium</option>
          <option value="large-v3">large-v3</option>
        </select>
      </div>
      <div>
        <label for="watermark">Watermark</label>
        <select id="watermark">
          <option value="mv-edits" selected>mv-edits</option>
          <option value="none">none</option>
        </select>
      </div>
    </div>
    <label for="ollama_model">Model that writes the copy (title, description, narration)</label>
    <input type="text" id="ollama_model" value="nemotron-3-super:cloud">
    <label for="editorial_file">Approved editorial file (optional, commentary or storytelling)</label>
    <input type="text" id="editorial_file" placeholder="Path to approved editorial.json">
    <label class="switch">
      <input type="checkbox" id="loudnorm" checked>
      <span class="sw-text">Loudness master (−14 LUFS)
        <span class="sw-sub">Normalize the final audio to the social-media standard so the clip is as loud and consistent as other creators'. On by default.</span>
      </span>
    </label>
    <label class="switch" id="silence-wrap">
      <input type="checkbox" id="remove_silence">
      <span class="sw-text">Tighten pauses (remove blank spaces)
        <span class="sw-sub">Cut the gaps where the audio is near-silent and remap the captions. Best on your own ranges; skipped for auto-picked recap beats.</span>
      </span>
    </label>
    <label class="switch">
      <input type="checkbox" id="mirror">
      <span class="sw-text">Mirror clip
        <span class="sw-sub">Flip the footage left-right; captions and narration stay upright.</span>
      </span>
    </label>
  </fieldset>

  <fieldset>
    <legend>⑨ DESCRIPTION &amp; CREDITS</legend>
    <label class="switch">
      <input type="checkbox" id="edit_credits">
      <span class="sw-text">Edit-style description
        <span class="sw-sub">Movie / Music / Editing credits, hashtags, and a fair-use disclaimer.</span>
      </span>
    </label>
    <div id="edit-credits-opts" style="display:none">
      <label for="music_credit">Music credit (the “Music:” line)</label>
      <input type="text" id="music_credit" placeholder="Artist – Track (auto-filled from the music file)">
      <label for="editing_program">Editing program (the “Editing Program:” line)</label>
      <input type="text" id="editing_program" value="Adobe After Effects 2024">
      <label class="switch">
        <input type="checkbox" id="edit_disclaimer" checked>
        <span class="sw-text">Include the © fair-use disclaimer</span>
      </label>
    </div>
  </fieldset>

  <details>
    <summary>Advanced</summary>
    <div class="row">
      <div>
        <label for="size">Square size (px)</label>
        <input type="number" id="size" value="1080">
      </div>
      <div>
        <label for="language">Language</label>
        <input type="text" id="language" value="en">
      </div>
    </div>
    <label class="switch">
      <input type="checkbox" id="skip_thumbnail">
      <span class="sw-text">Don't make a thumbnail at all
        <span class="sw-sub">Skips the thumbnail entirely. Leave off unless you already have one.</span>
      </span>
    </label>
  </details>

  <fieldset>
    <legend>COMPARE SHORT AND LONG EDITS</legend>
    <p>Use the enabled source window above. Both versions cover the same four story developments at the same runtime. The Short keeps dialogue with brief bylines; the landscape version uses fuller narration.</p>
    <label for="comparison-duration">Shared runtime (seconds)</label>
    <input id="comparison-duration" type="number" min="30" max="150" value="60" step="10">
    <label for="comparison-feedback">Review feedback (optional)</label><textarea id="comparison-feedback" placeholder="Corrections to character attribution or story continuity"></textarea>
<button type="button" id="comparison-build">Build and review both edit plans</button>
    <div id="comparison-review" style="white-space:pre-wrap"></div>
    <label><input type="checkbox" id="comparison-approved"> I reviewed character attribution, story continuity and the candidate frames.</label>
    <button type="button" id="comparison-short" disabled>Render Short — cuts and bylines</button>
    <button type="button" id="comparison-long" disabled>Render landscape — full narration</button>
  </fieldset>
  <button type="submit" id="submit">① Cut clip &amp; pick thumbnail</button>
  <div id="submit-hint"></div>
</form>

<div id="error"></div>

<div id="progress">
  <div id="step-label"></div>
  <div class="steps" id="steps"></div>
  <div id="log"></div>
</div>

<div id="result">
  <h2 id="result-title"></h2>
  <video id="result-video" controls></video>
  <div>
    <a id="result-desc" href="#" target="_blank">Description</a>
    <a id="result-meta" href="#" target="_blank">Metadata JSON</a>
    <a id="result-narration" href="#" target="_blank" style="display:none">Narration script</a>
  </div>

  <div id="thumb-editor" style="display:none">
    <div class="thumb-head">Thumbnail</div>
    <img id="result-thumb" style="display:none" alt="thumbnail">
    <div class="thumb-hint">Scrub to a frame, then move the crop slider to choose which slice of the shot to keep (the highlighted box shows it, sized to your output shape). Title and movie name are drawn on automatically.</div>
    <div id="thumb-video-wrap">
      <video id="thumb-video" controls preload="metadata"></video>
      <div id="thumb-cropbox"></div>
    </div>
    <label for="thumb-pan">Crop position <span id="thumb-pan-val">center</span></label>
    <input type="range" id="thumb-pan" min="0" max="100" value="50">
    <div class="row">
      <div><label for="thumb-title">Title (top of thumbnail + burned into the video)</label><input type="text" id="thumb-title" autocapitalize="off" autocorrect="off" spellcheck="false"></div>
      <div><label for="thumb-movie">Movie name (bottom)</label><input type="text" id="thumb-movie" autocapitalize="off" autocorrect="off" spellcheck="false"></div>
    </div>
    <button type="button" id="title-suggest" class="mini-btn" style="margin-top:8px">✨ Suggest titles</button>
    <div id="title-options"></div>
    <div class="row">
      <div>
        <label for="thumb-font">Font</label>
        <select id="thumb-font">
          <option value="impact" selected>Impact (classic)</option>
          <option value="cooper-black">Cooper Black (chunky retro)</option>
          <option value="showcard">Showcard Gothic (showbiz)</option>
          <option value="bernard">Bernard Condensed (poster)</option>
          <option value="bauhaus">Bauhaus 93 (funky 70s)</option>
          <option value="broadway">Broadway (art deco)</option>
          <option value="stencil">Stencil (military)</option>
          <option value="rockwell">Rockwell (slab serif)</option>
          <option value="bahnschrift">Bahnschrift (clean)</option>
          <option value="segoe-black">Segoe Black</option>
          <option value="arial-black">Arial Black</option>
          <option value="georgia">Georgia Bold (elegant)</option>
        </select>
      </div>
      <div>
        <label for="thumb-color">Text color</label>
        <select id="thumb-color">
          <option value="white" selected>White</option>
          <option value="gold">Gold</option>
          <option value="red">Red</option>
          <option value="orange">Orange</option>
          <option value="cyan">Cyan</option>
          <option value="lime">Lime</option>
          <option value="pink">Hot pink</option>
        </select>
      </div>
    </div>
    <label for="thumb-enhance">Image enhancement</label>
    <select id="thumb-enhance">
      <option value="subtle" selected>Subtle — light sharpen + colour pop</option>
      <option value="punchy">Punchy — brighter, bold colour, crisp</option>
      <option value="cinematic">Cinematic — deep contrast, cooler colour</option>
      <option value="bright">Bright — lift dark night scenes</option>
      <option value="none">None — untouched frame</option>
    </select>
    <details id="thumb-enhance-adv">
      <summary>Fine-tune brightness / contrast / saturation / sharpness</summary>
      <label for="thumb-brightness">Brightness <span id="thumb-brightness-val"></span></label>
      <input type="range" id="thumb-brightness" min="-0.4" max="0.4" step="0.01">
      <label for="thumb-contrast">Contrast <span id="thumb-contrast-val"></span></label>
      <input type="range" id="thumb-contrast" min="0.5" max="2" step="0.01">
      <label for="thumb-saturation">Saturation <span id="thumb-saturation-val"></span></label>
      <input type="range" id="thumb-saturation" min="0" max="2.5" step="0.01">
      <label for="thumb-sharpness">Sharpness <span id="thumb-sharpness-val"></span></label>
      <input type="range" id="thumb-sharpness" min="0" max="2" step="0.05">
      <button type="button" id="thumb-enhance-reset" class="mini-btn" style="margin-top:8px">↺ Back to preset</button>
    </details>
    <button type="button" id="thumb-grab" class="mini-btn" style="margin-top:8px">📸 Use this frame as thumbnail</button>
    <span id="thumb-status" class="thumb-status"></span>
    <button type="button" id="thumb-render" style="display:none">② Render final video →</button>
  </div>
</div>

<script>
const BASE_PATH  = '__BASE_PATH__';
const rangesEl   = document.getElementById('ranges');
const bgMode     = document.getElementById('bg_mode');
const bgFileWrap = document.getElementById('bg-file-wrap');
const bgFileSel  = document.getElementById('bg_music');
const bgVolume   = document.getElementById('bg_volume');
const volumeVal  = document.getElementById('volume-val');
const form       = document.getElementById('form');
const submitBtn  = document.getElementById('submit');
const errorBox   = document.getElementById('error');
const progressEl = document.getElementById('progress');
const logEl      = document.getElementById('log');
const resultEl   = document.getElementById('result');
const videoInput = document.getElementById('video');
const checkSrtBtn = document.getElementById('check-srt');
const srtStatus  = document.getElementById('srt-status');
const suggestWrap = document.getElementById('suggest-wrap');
const subsNone   = document.getElementById('subs-none');
const aiBtn      = document.getElementById('ai-btn');
const aiHint     = document.getElementById('ai-hint');
const aiStatus   = document.getElementById('ai-status');
const fmtCards   = document.getElementById('fmt-cards');
const aspectSel  = document.getElementById('aspect');
const aspectNote = document.getElementById('aspect-note');
const wantThumbChk = document.getElementById('want_thumbnail');
const thumbShapeNote = document.getElementById('thumb-shape-note');
const targetLen  = document.getElementById('target_len');
const targetLenVal = document.getElementById('target-len-val');
const lengthHint = document.getElementById('length-hint');
const rangesTotal = document.getElementById('ranges-total');
const rangesNote = document.getElementById('ranges-note');
const whisperWrap = document.getElementById('whisper-wrap');
const silenceWrap = document.getElementById('silence-wrap');
const musicNote  = document.getElementById('music-note');
const submitHint = document.getElementById('submit-hint');
const selSelection = document.getElementById('sel_selection');
const selVoice     = document.getElementById('sel_voice');
const axisNote     = document.getElementById('axis-note');
const axesDetails  = document.getElementById('axes-details');
const narrateLevelWrap  = document.getElementById('narrate-level-wrap');
const dialogueLevelWrap = document.getElementById('dialogue-level-wrap');
const narrateKind       = document.getElementById('narrate-kind');
const scriptFieldset    = document.getElementById('script-fieldset');
const rangesWrap        = document.getElementById('ranges-wrap');
const dialogueVol       = document.getElementById('dialogue_volume');
const dialogueVolVal    = document.getElementById('dvolume-val');
const nVolLabel         = document.getElementById('nvol-label');
const suggestModel = document.getElementById('suggest_model');
const useSrtWrap = document.getElementById('use-srt-wrap');
const useSrtChk  = document.getElementById('use_srt');
const suggestionsEl = document.getElementById('suggestions');
const stepsEl    = document.getElementById('steps');
const stepLabel  = document.getElementById('step-label');
const narrateChk = document.getElementById('narrate');
const narrateOpts = document.getElementById('narrate-opts');
const nVolume    = document.getElementById('narration_volume');
const nVolumeVal = document.getElementById('nvolume-val');
const formatSel  = document.getElementById('format');
const recapFieldset = document.getElementById('recap-fieldset');
const narrateFieldset = document.getElementById('narrate-fieldset');

let probedSrt = null;
// Real total comes from /run (and is confirmed by the first '[n/N]' log line).
// Never hardcode it — it varies with narration / thumbnail baking / silence removal.
let totalSteps = 0;

function addRangeRow(start, end) {
  const row = document.createElement('div');
  row.className = 'range-row';
  row.innerHTML = `
    <input type="text" class="r-start" placeholder="3:20" value="${start||''}">
    <span style="color:var(--text-faint)">–</span>
    <input type="text" class="r-end" placeholder="3:30" value="${end||''}">
    <button type="button" class="mini-btn r-remove">✕</button>
  `;
  row.querySelector('.r-remove').onclick = () => {
    if (rangesEl.children.length > 1) row.remove();
  };
  rangesEl.appendChild(row);
}
addRangeRow();
document.getElementById('add-range').onclick = () => addRangeRow();

const bgAudioOpts = document.getElementById('bg-audio-opts');
function syncBgMode() {
  bgFileWrap.style.display = bgMode.value === 'file' ? 'block' : 'none';
  // Volume + skip-intro only matter when background music is actually on.
  bgAudioOpts.style.display = bgMode.value === 'none' ? 'none' : 'block';
}
bgMode.addEventListener('change', () => { syncBgMode(); syncMusicNote(); });
syncBgMode();
let bgVolTouched = false;
bgVolume.addEventListener('input', () => { bgVolTouched = true; volumeVal.textContent = bgVolume.value; });
let nVolTouched = false;
nVolume.addEventListener('input', () => { nVolTouched = true; nVolumeVal.textContent = nVolume.value; });
dialogueVol.addEventListener('input', () => { dialogueVolVal.textContent = dialogueVol.value; });

// ── Output shape drives geometry, clip length, and the thumbnail canvas ──────
// It is one decision, not three: a 9:16 Short and a 16:9 long-form upload want
// different clip lengths AND differently-shaped thumbnails.
const SHAPES = {
  vertical:  {name: '9:16 vertical',  thumb: [1080, 1920], short: true},
  square:    {name: '1:1 square',     thumb: [1080, 1080], short: true},
  landscape: {name: '16:9 landscape', thumb: [1920, 1080], short: false},
};
const SHORT_CAP = 180;   // YouTube stops calling it a Short past three minutes

// A preset is a named POINT in the selection x voice x shape space, not a type.
// Everything below branches on the axes; nothing branches on the preset name.
const PRESETS = {
  narrated:     {selection: 'manual', voice: 'commentary',   shape: 'vertical'},
  storytelling: {selection: 'script', voice: 'storytelling', shape: 'landscape'},
  recap:        {selection: 'spine',  voice: 'music',        shape: 'vertical'},
  classic:      {selection: 'manual', voice: 'none',         shape: 'square'},
};
const sel   = () => selSelection.value;
const voice = () => selVoice.value;
// Derived facts — the same ones main.py computes.
const narrationLed = () => voice() === 'storytelling';   // script sets clip length
const autoRanges   = () => sel() === 'spine' || sel() === 'script';
const needsSubs    = () => autoRanges() || sel() === 'scene';

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  selSelection.value = p.selection;
  selVoice.value = p.voice;
  aspectSel.value = p.shape;
}
// Which preset (if any) the current axes spell out — so hand-tuning the axes
// lights up the matching card instead of pretending nothing changed.
function matchPreset() {
  for (const [name, p] of Object.entries(PRESETS)) {
    if (p.selection === sel() && p.voice === voice() && p.shape === aspectSel.value) return name;
  }
  return 'custom';
}

function shapeKey() {
  return aspectSel.value || 'vertical';
}
function shapeInfo() { return SHAPES[shapeKey()] || SHAPES.vertical; }

function fmtDur(sec) {
  sec = Math.round(sec);
  if (sec < 60) return sec + 's';
  return Math.floor(sec / 60) + ':' + String(sec % 60).padStart(2, '0');
}

// ── Target length ───────────────────────────────────────────────────────────
// One control, two meanings: for a single-scene edit it is the length the scene
// AI aims for; for a recap it is the total body length before the outro. The
// bounds follow the output shape — minutes for long-form, seconds for a Short.
function lengthBounds() {
  const long = !shapeInfo().short;
  const recap = autoRanges();
  if (long)  return {min: 120, max: 900, step: 30, def: recap ? 420 : 300};
  if (recap) return {min: 30,  max: 300, step: 10, def: 120};
  return {min: 20, max: 180, step: 5, def: 60};
}
function sceneWindow() {
  // The window handed to the scene AI, either side of the target.
  const v = parseFloat(targetLen.value) || lengthBounds().def;
  return [Math.max(5, Math.round(v * 0.7)), Math.round(v * 1.4)];
}
let lengthTouched = false;
function syncLength() {
  const b = lengthBounds();
  targetLen.min = b.min; targetLen.max = b.max; targetLen.step = b.step;
  let v = lengthTouched ? parseFloat(targetLen.value) : b.def;
  if (!isFinite(v)) v = b.def;
  v = Math.min(b.max, Math.max(b.min, Math.round(v / b.step) * b.step));
  targetLen.value = v;
  targetLenVal.textContent = fmtDur(v);
  const w = sceneWindow();
  // Both ends of a range share one unit — "42s–1:24" reads as a typo.
  const mmss = (t) => Math.floor(t / 60) + ':' + String(Math.round(t) % 60).padStart(2, '0');
  const pair = Math.max(w[0], w[1]) >= 60
    ? mmss(w[0]) + '–' + mmss(w[1])
    : w[0] + '–' + w[1] + 's';
  lengthHint.innerHTML = autoRanges()
    ? 'The recap body aims for <strong>' + fmtDur(v) + '</strong> of cuts before the outro.'
    : 'Scene suggestions will aim for <strong>' + pair + '</strong>.';
}
targetLen.addEventListener('input', () => { lengthTouched = true; syncLength(); });

// ── Live total of the picked ranges ─────────────────────────────────────────
function syncRangesTotal() {
  let total = 0, n = 0;
  for (const row of rangesEl.children) {
    const s = parseTs(row.querySelector('.r-start').value);
    const e = parseTs(row.querySelector('.r-end').value);
    if (s == null || e == null || e <= s) continue;
    total += e - s; n++;
  }
  const over = shapeInfo().short && total > SHORT_CAP;
  rangesTotal.textContent = n ? (n + (n > 1 ? ' clips · ' : ' clip · ') + fmtDur(total)) : '';
  rangesTotal.classList.toggle('over', over);
  const notes = [];
  if (autoRanges() && !n) {
    notes.push('Empty = recap the whole film. One long range scopes it to that stretch.');
  }
  if (over) {
    notes.push('Past the ' + fmtDur(SHORT_CAP) + ' Shorts limit — YouTube will treat this as a regular video.');
  }
  rangesNote.innerHTML = notes.join('<br>');
}
rangesEl.addEventListener('input', syncRangesTotal);

// ── Music note: what a track does (or does not do) in this mode ─────────────
function syncMusicNote() {
  const v = voice(), m = bgMode.value;
  let msg = '', wantVol = '0.10';
  if (v === 'music') {
    msg = m === 'none'
      ? 'Nothing is carrying this edit — pick a track, or switch the voice above to dialogue or a narrator.'
      : 'The music leads: cuts snap to its beats and the dialogue ducks underneath it.';
    if (m !== 'none') wantVol = '0.60';
  } else if (v === 'storytelling') {
    msg = m === 'none'
      ? 'The narrator leads and the dialogue sits under them. Music is optional here — a quiet bed is typical.'
      : 'Keep this low: it sits under the narrator alongside the dialogue.';
    if (m !== 'none') wantVol = '0.10';
  } else if (v === 'commentary' && m !== 'none') {
    msg = 'Movie audio + voiceover + music gets muddy fast — keep this low.';
  }
  // Only move the slider while the user hasn't touched it, so a hand-set level sticks.
  if (!bgVolTouched && bgVolume.value !== wantVol) {
    bgVolume.value = wantVol;
    volumeVal.textContent = bgVolume.value;
  }
  musicNote.innerHTML = msg;
}

// ── Axis sync: every widget follows the three axes, never a preset name ─────
function syncFormat() {
  const s = sel(), v = voice(), led = narrationLed();

  // Light the card whose point in the space this is (or none, when custom).
  const preset = matchPreset();
  formatSel.value = preset;
  for (const card of fmtCards.querySelectorAll('.fmt')) {
    card.setAttribute('aria-pressed', String(card.dataset.value === preset));
  }
  if (preset === 'custom') axesDetails.open = true;

  // Voice decides which narration controls apply. Storytelling and commentary
  // are different agents with different mixes, so they get different copy.
  const anyVoice = v === 'commentary' || v === 'storytelling';
  document.getElementById('audio-mix-wrap').style.display = v === 'commentary' ? '' : 'none';
  const mixLed = led || (v === 'commentary' && document.getElementById('audio_mix').value !== 'original');
  narrateFieldset.style.display = anyVoice ? '' : 'none';
  narrateChk.checked = v === 'commentary';   // legacy flag main.py still reads
  narrateLevelWrap.style.display = anyVoice ? '' : 'none';
  // Commentary sits UNDER a film at full volume; a storytelling narrator IS the
  // lead. Same control, two different working levels.
  nVolLabel.textContent = led ? 'Narrator level' : 'Voiceover volume';
  if (!nVolTouched) {
    nVolume.value = mixLed ? '1' : '0.04';
    nVolumeVal.textContent = parseFloat(nVolume.value).toFixed(2);
  }
  dialogueLevelWrap.style.display = mixLed ? '' : 'none';
  narrateKind.innerHTML = mixLed
    ? 'The narrator is the <strong>lead track</strong> and retells the plot; the film’s dialogue is pulled down and ducks under them. Their lines render as a lower third.'
    : 'The voiceover sits <strong>under</strong> the film at low volume and comments on the moment. Its text shows in the band beneath the video.';

  // Selection decides which agent picks the clips.
  recapFieldset.style.display = s === 'spine' ? '' : 'none';
  scriptFieldset.style.display = s === 'script' ? '' : 'none';
  rangesWrap.style.display = s === 'script' ? 'none' : '';

  // Whisper only ever runs without usable subtitles, and every selection agent
  // needs the subtitle file outright.
  whisperWrap.style.display = (needsSubs() || (probedSrt && useSrtChk.checked)) ? 'none' : '';
  // Silence removal would desync agent-picked cuts from their captions.
  silenceWrap.style.display = autoRanges() ? 'none' : '';
  // Edit-style credits are the norm for multi-clip agent edits.
  if (autoRanges() && !editCreditsTouched && !editCreditsChk.checked) {
    editCreditsChk.checked = true;
    syncEditCredits();
  }

  aiBtn.textContent = s === 'spine'
    ? '🎬 Build recap beats from subtitles'
    : '✨ Suggest scenes from subtitles';
  aiHint.innerHTML = s === 'spine'
    ? 'The whole plot goes to the model; it comes back as an ordered beat list, loaded into the ranges above. Can take a minute.'
    : 'The whole subtitle file goes to the model; it comes back with clip ranges you can click to add.';
  aiBtn.style.display = s === 'script' ? 'none' : '';
  if (s === 'script') {
    aiHint.innerHTML = 'The script itself picks the clips at render time — there is nothing to preview here yet.';
  }

  axisNote.innerHTML = led
    ? 'Narration-led: the script is written and spoken first, then each clip is cut to last exactly as long as its line.'
    : 'Clip-led: the ranges fix the length, and anything spoken is fitted inside them.';

  syncShape();
  syncLength();
  syncMusicNote();
  syncRangesTotal();
  submitLabel();
}

// ── Shape + thumbnail notes ────────────────────────────────────────────────
let cropTargetAspect = 9 / 16;   // read by the thumbnail crop-box, further down
let cropReady = false;
function syncShape() {
  const sh = shapeInfo();
  const w = sh.thumb[0], h = sh.thumb[1];
  // Shape is free in every combination now — the only thing worth saying is what
  // the choice implies for length and for the thumbnail.
  aspectNote.classList.remove('lock');
  aspectNote.innerHTML = sh.short
    ? 'Short-form: stay under ' + fmtDur(SHORT_CAP) + ' to keep it a Short.'
    : 'Long-form: no ' + fmtDur(SHORT_CAP) + ' limit, and the thumbnail is composed 16:9.';
  if (voice() === 'commentary' && !sh.short) {
    aspectNote.innerHTML += '<br>Commentary text renders as a lower third here — there is no band to put it in.';
  }

  // Narration-led rendering can't be cut ahead of time: the clips don't exist
  // until the script has been spoken and measured.
  const canPick = !narrationLed();
  wantThumbChk.disabled = !canPick;
  thumbShapeNote.innerHTML = !canPick
    ? 'Not available for a narration-led edit — the clips don’t exist until the script has been spoken. A <strong>'
      + w + '×' + h + '</strong> thumbnail is composed automatically.'
    : (wantThumbChk.checked
        ? 'Cut first, scrub to the frame you want, then render. Composed at <strong>'
          + w + '×' + h + '</strong> for the ' + sh.name + ' output.'
        : 'Off — renders in a single pass and composes a <strong>' + w + '×' + h
          + '</strong> thumbnail automatically.');
  cropTargetAspect = w / h;
  refreshCropbox();
}
wantThumbChk.addEventListener('change', () => { syncShape(); submitLabel(); });
aspectSel.addEventListener('change', () => { syncFormat(); });

// ── Cards set all three axes at once; the axes themselves stay editable ─────
fmtCards.addEventListener('click', (e) => {
  const card = e.target.closest('.fmt');
  if (!card) return;
  applyPreset(card.dataset.value);
  syncFormat();
});
selSelection.addEventListener('change', syncFormat);
selVoice.addEventListener('change', syncFormat);
document.getElementById('audio_mix').addEventListener('change', syncFormat);

// ── Edit-style description credits ──────────────────────────────────────────
const editCreditsChk = document.getElementById('edit_credits');
const editCreditsOpts = document.getElementById('edit-credits-opts');
const musicCreditInput = document.getElementById('music_credit');
let editCreditsTouched = false;
function cleanMusicName(fn) {
  return (fn || '').replace(/\\.[A-Za-z0-9]+$/, '').replace(/^\\s*\\d+[\\.\\)\\-\\s]+/, '')
    .replace(/_/g, ' ').replace(/\\s+/g, ' ').trim();
}
function syncEditCredits() {
  editCreditsOpts.style.display = editCreditsChk.checked ? 'block' : 'none';
  // Prefill the music credit from the chosen track, but never clobber a manual edit.
  if (editCreditsChk.checked && !musicCreditInput.value.trim() && bgMode.value === 'file' && bgFileSel.value) {
    musicCreditInput.value = cleanMusicName(bgFileSel.value);
  }
}
editCreditsChk.addEventListener('change', () => { editCreditsTouched = true; syncEditCredits(); });
bgFileSel.addEventListener('change', () => {
  if (editCreditsChk.checked && !musicCreditInput.value.trim()) {
    musicCreditInput.value = cleanMusicName(bgFileSel.value);
  }
});

// ── Recap outro ────────────────────────────────────────────────────────────
const outroSec = document.getElementById('outro_seconds');
const outroSecVal = document.getElementById('outro-sec-val');
outroSec.addEventListener('input', () => { outroSecVal.textContent = outroSec.value + 's'; });
const recapOutroChk = document.getElementById('recap_outro');
recapOutroChk.addEventListener('change', () => {
  document.getElementById('recap-outro-opts').style.display = recapOutroChk.checked ? 'block' : 'none';
});

// ── One AI button, whichever edit you are cutting ───────────────────────────
// Both jobs read the same subtitle file with the same model; only the shape of
// the answer differs, so they belong on the same control.
function aiBusy(on, label) {
  aiBtn.disabled = on;
  aiBtn.textContent = on ? label : (sel() === 'spine'
    ? '🎬 Build recap beats from subtitles'
    : '✨ Suggest scenes from subtitles');
}
function aiSay(msg, bad) {
  aiStatus.style.color = bad ? 'var(--err)' : 'var(--text-faint)';
  aiStatus.textContent = msg || '';
}
function subtitleModel() {
  return suggestModel.value === '__cloud__'
    ? document.getElementById('ollama_model').value.trim()
    : suggestModel.value;
}

async function runSceneSuggest() {
  const w = sceneWindow();
  curatedRegion = null;   // a fresh set of scenes invalidates any prior curation
  aiBusy(true, '⏳ Reading the subtitles…');
  aiSay('');
  suggestionsEl.innerHTML = '';
  try {
    const res = await fetch(BASE_PATH + '/suggest-scenes', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        srt: probedSrt,
        title: document.getElementById('source_title').value.trim(),
        model: subtitleModel(),
        count: 5,
        min_seconds: w[0],
        max_seconds: w[1],
      }),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    const d = await res.json();
    renderSuggestions(d.scenes || []);
  } catch (err) {
    suggestionsEl.innerHTML = '<div class="suggestion" style="cursor:default;color:var(--err)">Suggestion failed: '
      + String(err.message).slice(0, 300) + '</div>';
  } finally {
    aiBusy(false);
  }
}

async function runRecapBuild() {
  aiBusy(true, '⏳ Reading the whole plot… (can take a minute)');
  aiSay('');
  try {
    const res = await fetch(BASE_PATH + '/build-recap', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        srt: probedSrt,
        title: document.getElementById('source_title').value.trim(),
        model: subtitleModel(),
        target_seconds: parseInt(targetLen.value, 10),
      }),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    const d = await res.json();
    const beats = d.beats || d.scenes || [];
    if (!beats.length) { aiSay('No recap beats came back — try a different model.', true); return; }
    rangesEl.innerHTML = '';
    for (const b of beats) {
      const [s, e] = (b.range || '').split('-');
      addRangeRow(s, e);
    }
    const total = beats.reduce((a, b) => a + (b.end - b.start), 0);
    syncRangesTotal();
    aiSay(beats.length + ' beats loaded (~' + fmtDur(total) + '). Tweak them above, then render.');
  } catch (err) {
    aiSay('Recap build failed: ' + String(err.message).slice(0, 300), true);
  } finally {
    aiBusy(false);
  }
}

aiBtn.addEventListener('click', () => {
  if (!(useSrtChk.checked && probedSrt)) {
    aiSay('Needs a subtitle file — none is selected for this movie.', true);
    return;
  }
  return sel() === 'spine' ? runRecapBuild() : runSceneSuggest();
});

// ── SRT probe ──────────────────────────────────────────────────────────────
// Auto-runs whenever the path settles, so the subtitle options are simply
// THERE when a subtitle file exists, in every mode — rather than hiding behind
// a button you had to know to press.
let probeSeq = 0;
let probeTimer = null;

function showNoSrt(msg, cls) {
  probedSrt = null;
  useSrtWrap.style.display = 'none';
  suggestWrap.style.display = 'none';
  subsNone.style.display = '';
  suggestionsEl.innerHTML = '';
  srtStatus.className = cls || '';
  srtStatus.textContent = msg;
  syncFormat();
}

async function probeVideo() {
  const seq = ++probeSeq;
  const video = videoInput.value.trim();
  suggestionsEl.innerHTML = '';
  if (!video) { showNoSrt('Paste a movie path — subtitles are looked up automatically.'); return; }
  srtStatus.className = '';
  srtStatus.textContent = 'Looking for a subtitle file…';
  checkSrtBtn.disabled = true;
  try {
    const res = await fetch(BASE_PATH + '/probe-video', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({video}),
    });
    const d = await res.json();
    if (seq !== probeSeq) return;   // a newer probe already answered
    if (!d.exists) { showNoSrt('⚠ No file at that path.', 'missing'); return; }
    if (typeof d.duration === 'number' && d.duration > 0) setSpanBounds(d.duration);
    if (d.srt) {
      probedSrt = d.srt;
      srtStatus.className = 'found';
      srtStatus.textContent = '✓ Subtitles found: ' + d.srt.split(/[\\\\/]/).pop();
      useSrtWrap.style.display = 'flex';
      suggestWrap.style.display = 'block';
      subsNone.style.display = 'none';
      syncFormat();
    } else {
      showNoSrt('No .srt next to the movie — Whisper will transcribe instead.', 'missing');
    }
  } catch (err) {
    if (seq !== probeSeq) return;
    showNoSrt('Could not check (' + String(err.message).slice(0, 120) + ')', 'missing');
  } finally {
    checkSrtBtn.disabled = false;
  }
}
useSrtChk.addEventListener('change', syncFormat);

// Some iOS paste targets deliver a copied path URL-encoded ('\\' as %5C,
// ' ' as %20). Decode in place so the user sees the real path immediately.
function normalizePathInput() {
  let v = videoInput.value.trim();
  if (/%(5C|20|3A|2F)/i.test(v)) {
    try { v = decodeURIComponent(v); } catch (err) {}
  }
  if (v.length >= 2 && v[0] === v[v.length - 1] && (v[0] === '"' || v[0] === "'")) {
    v = v.slice(1, -1);
  }
  if (v !== videoInput.value) videoInput.value = v;
}
videoInput.addEventListener('input', () => {
  normalizePathInput();
  clearTimeout(probeTimer);
  probeTimer = setTimeout(probeVideo, 600);
});
videoInput.addEventListener('paste', () => setTimeout(() => {
  normalizePathInput();
  clearTimeout(probeTimer);
  probeTimer = setTimeout(probeVideo, 150);
}, 0));
videoInput.addEventListener('change', () => { clearTimeout(probeTimer); probeVideo(); });
checkSrtBtn.addEventListener('click', () => { clearTimeout(probeTimer); probeVideo(); });

function renderSuggestions(scenes) {
  suggestionsEl.innerHTML = '';
  if (!scenes.length) {
    suggestionsEl.innerHTML = '<div class="suggestion" style="cursor:default">No scenes suggested.</div>';
    return;
  }
  for (const sc of scenes) {
    const card = document.createElement('div');
    card.className = 'suggestion';
    card.dataset.range = sc.range;
    card.innerHTML = `
      <div class="s-head">
        <label style="display:flex;gap:8px;align-items:center;cursor:pointer;min-width:0">
          <input type="checkbox" class="s-tick" style="accent-color:var(--accent);flex-shrink:0">
          <span style="overflow:hidden;text-overflow:ellipsis">${sc.title || 'Scene'}</span>
        </label>
        <span class="s-range">${sc.range}</span>
      </div>
      <div class="s-reason">${sc.reason || ''}</div>
      <button type="button" class="mini-btn s-add" style="margin-top:6px">+ add as clip range</button>
    `;
    // Ticking a scene marks it for the curated recap region; the button adds it
    // as a literal clip range (the old behaviour).
    card.querySelector('.s-tick').addEventListener('change', updateCurationBar);
    card.querySelector('.s-add').onclick = () => {
      const [s, e] = sc.range.split('-');
      const empty = [...rangesEl.children].find(r =>
        !r.querySelector('.r-start').value.trim() && !r.querySelector('.r-end').value.trim());
      if (empty) { empty.querySelector('.r-start').value = s; empty.querySelector('.r-end').value = e; }
      else { addRangeRow(s, e); }
      card.classList.add('added');
    };
    suggestionsEl.appendChild(card);
  }
  // Curation bar: tick scenes → build the recap within just those.
  const bar = document.createElement('div');
  bar.id = 'curation-bar';
  bar.style = 'margin-top:10px';
  bar.innerHTML = `
    <button type="button" id="use-ticked" class="mini-btn primary" style="width:100%">🎬 Use ticked scenes as the recap region</button>
    <div id="curation-note" class="note"></div>`;
  suggestionsEl.appendChild(bar);
  bar.querySelector('#use-ticked').onclick = useTickedScenes;
  updateCurationBar();
}

function tickedSceneRanges() {
  return [...suggestionsEl.querySelectorAll('.suggestion')]
    .filter(c => { const t = c.querySelector('.s-tick'); return t && t.checked; })
    .map(c => c.dataset.range);
}
function updateCurationBar() {
  const n = tickedSceneRanges().length;
  const note = document.getElementById('curation-note');
  if (note) note.textContent = n
    ? `${n} scene(s) ticked — the recap will pick connecting beats within only these.`
    : 'Tick the scenes you like, then build the recap within just them.';
}
// Ticked scenes → the recap region. Sets Recap format so the picker runs, and
// stores the region for buildPayload; the user then sets target + hits Build.
let curatedRegion = null;
function useTickedScenes() {
  const ranges = tickedSceneRanges();
  if (!ranges.length) { updateCurationBar(); return; }
  curatedRegion = ranges.join('; ');
  const recapCard = fmtCards.querySelector('.fmt[data-value="recap"]');
  if (recapCard) recapCard.click();          // switch to the recap (spine) preset
  const note = document.getElementById('curation-note');
  if (note) note.innerHTML = `<strong>Recap region set to ${ranges.length} ticked scene(s).</strong> `
    + 'Set the target length, then hit Build — it will connect beats within only these.';
}

fetch(BASE_PATH + '/audio-files').then(r => r.json()).then(files => {
  for (const f of files) {
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = f;
    bgFileSel.appendChild(opt);
  }
});

function collectRanges() {
  const parts = [];
  for (const row of rangesEl.children) {
    const s = row.querySelector('.r-start').value.trim();
    const e = row.querySelector('.r-end').value.trim();
    if (s && e) parts.push(`${s}-${e}`);
  }
  return parts.join('; ');
}

function buildSteps(total) {
  if (!total || total === totalSteps) return;
  totalSteps = total;
  stepsEl.innerHTML = '';
  for (let i = 1; i <= total; i++) {
    const div = document.createElement('div');
    div.className = 'step';
    div.dataset.n = i;
    stepsEl.appendChild(div);
  }
}

function setStep(n) {
  document.querySelectorAll('.step').forEach(el => {
    const i = parseInt(el.dataset.n, 10);
    el.classList.toggle('past', i < n);
    el.classList.toggle('active', i === n);
  });
}

function appendLog(text) {
  logEl.textContent += text + '\\n';
  logEl.scrollTop = logEl.scrollHeight;
}

let wsHandledDone = false;
function connectWs() {
  // wss:// when the page is served over https (e.g. behind Agent Hub's proxy),
  // otherwise ws://. A hardcoded ws:// silently fails on an https origin.
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}${BASE_PATH}/ws`);
  // On (re)connect the server replays the full backlog, so start the log clean
  // to avoid duplicating lines after a dropped-and-restored connection.
  ws.onopen = () => { logEl.textContent = ''; };
  ws.onmessage = ev => {
    const d = JSON.parse(ev.data);
    if (d.type === 'line') {
      appendLog(d.text);
      buildSteps(d.total);
      if (d.step) {
        setStep(d.step);
        stepLabel.className = '';
        stepLabel.textContent = `Step ${d.step}/${d.total}` + (d.step_text ? ` — ${d.step_text}` : '');
      }
    } else if (d.type === 'done') {
      wsHandledDone = true;
      submitBtn.disabled = false;
      submitLabel();
      const renderBtn = document.getElementById('thumb-render');
      renderBtn.disabled = false;
      if (d.ok) {
        renderBtn.style.display = 'none';   // phase-2 done — thumbnail already baked in
        setStep(totalSteps + 1);
        stepLabel.className = 'done';
        stepLabel.textContent = `✓ Completed — all ${totalSteps} steps`;
        showResult(d.result);
      } else {
        stepLabel.className = 'failed';
        stepLabel.textContent = stepLabel.textContent
          ? stepLabel.textContent.replace(/^Step /, '✗ Failed at step ') : '✗ Failed';
        errorBox.style.display = 'block';
        errorBox.textContent = 'Pipeline failed — see log above.';
      }
    }
  };
  // Reconnect on drop (proxy timeout, network blip) so a completed run that
  // finished while we were disconnected still surfaces its result on reconnect.
  ws.onclose = () => { setTimeout(connectWs, 2000); };
}
connectWs();

let rawClipUrl = null;
let preparedDir = null;    // clip dir from /prepare — sent back with /run
let preparedRanges = null; // the ranges that dir was cut from — /run refuses a mismatch
let thumbGrabbed = false;  // has the user grabbed a frame since the last prepare?
function showResult(result) {
  resultEl.style.display = 'block';
  document.getElementById('result-title').textContent = result.title || 'Clip ready';
  const video = document.getElementById('result-video');
  if (result.video_url) { video.src = result.video_url; video.style.display = 'block'; }
  // Visibility set explicitly BOTH ways — a hidden link from a previous run
  // must come back when the next run has that artifact.
  const descA = document.getElementById('result-desc');
  const metaA = document.getElementById('result-meta');
  if (result.description_url) { descA.href = result.description_url; descA.style.display = 'inline-block'; }
  else { descA.style.display = 'none'; }
  if (result.metadata_url) { metaA.href = result.metadata_url; metaA.style.display = 'inline-block'; }
  else { metaA.style.display = 'none'; }
  const thumb = document.getElementById('result-thumb');
  if (result.thumbnail_url) { thumb.src = result.thumbnail_url; thumb.style.display = 'block'; }
  else { thumb.style.display = 'none'; }
  const narrA = document.getElementById('result-narration');
  if (result.narration_url) { narrA.href = result.narration_url; narrA.style.display = 'inline-block'; }
  else { narrA.style.display = 'none'; }

  // Thumbnail frame picker — only when we have the raw (uncropped) clip.
  const editor = document.getElementById('thumb-editor');
  rawClipUrl = result.raw_clip_url || null;
  if (rawClipUrl) {
    thumbVideo.src = rawClipUrl;
    document.getElementById('thumb-title').value = result.title || '';
    document.getElementById('thumb-movie').value = result.movie || '';
    editor.style.display = 'block';
  } else {
    editor.style.display = 'none';
  }
}

// ── Thumbnail crop-window (pan) picker ──────────────────────────────────────
const thumbVideo = document.getElementById('thumb-video');
const thumbCropbox = document.getElementById('thumb-cropbox');
const thumbPan = document.getElementById('thumb-pan');
const thumbPanVal = document.getElementById('thumb-pan-val');
let cropWidthFrac = 1;   // width of the crop window as a fraction of the frame width

// Recompute the crop window whenever either input changes: the source frame's
// aspect, or the target thumbnail's (which follows the output shape — a 16:9
// long-form thumbnail crops almost nothing, a 9:16 Short crops a tall slice).
function refreshCropbox() {
  if (!cropReady || !thumbVideo.videoWidth) return;
  const srcAspect = thumbVideo.videoWidth / thumbVideo.videoHeight;
  cropWidthFrac = Math.min(1, cropTargetAspect / srcAspect);
  updateCropbox();
}

function updateCropbox() {
  const pan = parseFloat(thumbPan.value) / 100;
  if (cropWidthFrac >= 1) {
    thumbCropbox.style.display = 'none';
  } else {
    thumbCropbox.style.display = 'block';
    thumbCropbox.style.width = (cropWidthFrac * 100) + '%';
    thumbCropbox.style.left = (pan * (1 - cropWidthFrac) * 100) + '%';
  }
  thumbPanVal.textContent = pan < 0.34 ? 'left' : pan > 0.66 ? 'right' : 'center';
}
thumbPan.addEventListener('input', updateCropbox);
thumbVideo.addEventListener('loadedmetadata', () => {
  cropReady = true;
  refreshCropbox();
});

// ── Title options ───────────────────────────────────────────────────────────
const titleSuggestBtn = document.getElementById('title-suggest');
const titleOptionsEl  = document.getElementById('title-options');

titleSuggestBtn.addEventListener('click', async () => {
  const ranges = collectRanges();
  if (!ranges) {
    titleOptionsEl.innerHTML = '<div class="suggestion" style="cursor:default;color:var(--err)">Add a clip range first.</div>';
    return;
  }
  titleSuggestBtn.disabled = true;
  titleSuggestBtn.textContent = '⏳ Writing title options…';
  titleOptionsEl.innerHTML = '';
  try {
    const res = await fetch(BASE_PATH + '/suggest-titles', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        // Only pass the subtitle when it's the one selected for captions —
        // otherwise let the server transcribe the cut clip so the titles
        // describe THIS clip, not an unrelated .srt.
        srt: (useSrtChk.checked && probedSrt) ? probedSrt : null,
        ranges,
        raw_clip: rawClipUrl,
        whisper_model: document.getElementById('whisper_model').value,
        language: document.getElementById('language').value.trim(),
        source_title: document.getElementById('source_title').value.trim()
          || document.getElementById('thumb-movie').value.trim(),
        model: document.getElementById('ollama_model').value.trim(),
        count: 5,
      }),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    const d = await res.json();
    renderTitleOptions(d.titles || []);
  } catch (err) {
    titleOptionsEl.innerHTML = '<div class="suggestion" style="cursor:default;color:var(--err)">'
      + String(err.message).slice(0, 300) + '</div>';
  } finally {
    titleSuggestBtn.disabled = false;
    titleSuggestBtn.textContent = '✨ Suggest titles';
  }
});

function renderTitleOptions(titles) {
  titleOptionsEl.innerHTML = '';
  if (!titles.length) {
    titleOptionsEl.innerHTML = '<div class="suggestion" style="cursor:default">No titles suggested.</div>';
    return;
  }
  for (const t of titles) {
    const card = document.createElement('div');
    card.className = 'suggestion';
    card.innerHTML = `<div class="s-head"><span>${t.title}</span>`
      + `<span class="s-range">${(t.title || '').length} ch</span></div>`
      + (t.angle ? `<div class="s-reason">${t.angle}</div>` : '');
    card.title = 'Click to use this title';
    card.onclick = () => {
      document.getElementById('thumb-title').value = t.title;
      [...titleOptionsEl.children].forEach(c => c.classList.remove('added'));
      card.classList.add('added');
    };
    titleOptionsEl.appendChild(card);
  }
}

// ── Thumbnail enhancement (preset + optional per-knob overrides) ────────────
const ENHANCE_KNOBS = ['brightness', 'contrast', 'saturation', 'sharpness'];
const enhanceSel = document.getElementById('thumb-enhance');
let enhancePresets = {};   // filled from /enhance-presets
let enhanceTouched = false; // has the user moved a slider off the preset?

function applyPresetToSliders() {
  const p = enhancePresets[enhanceSel.value];
  if (!p) return;
  for (const k of ENHANCE_KNOBS) {
    const el = document.getElementById('thumb-' + k);
    el.value = p[k];
    document.getElementById('thumb-' + k + '-val').textContent = Number(p[k]).toFixed(2);
  }
  enhanceTouched = false;
}

fetch(BASE_PATH + '/enhance-presets').then(r => r.json()).then(d => {
  enhancePresets = d.presets || {};
  if (d.default) enhanceSel.value = d.default;
  applyPresetToSliders();
}).catch(() => {});

enhanceSel.addEventListener('change', applyPresetToSliders);
for (const k of ENHANCE_KNOBS) {
  document.getElementById('thumb-' + k).addEventListener('input', (e) => {
    document.getElementById('thumb-' + k + '-val').textContent = Number(e.target.value).toFixed(2);
    enhanceTouched = true;
  });
}
document.getElementById('thumb-enhance-reset').addEventListener('click', applyPresetToSliders);

// Send the raw knob values only once the user has actually moved one — an
// untouched panel means "just use the preset", and echoing the preset's own
// numbers back as overrides would freeze the thumbnail on stale values if the
// preset table ever changes.
function enhancePayload() {
  const out = {enhance: enhanceSel.value};
  if (enhanceTouched) {
    for (const k of ENHANCE_KNOBS) {
      out[k] = parseFloat(document.getElementById('thumb-' + k).value);
    }
  }
  return out;
}

function parseTs(s) {
  const p = String(s || '').trim().split(':').map(Number);
  if (!p.length || p.some(isNaN)) return null;
  return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2]
       : p.length === 2 ? p[0] * 60 + p[1] : p[0];
}

// Map a time on the cut clip's timeline back to the source movie's timeline
// (the clip is the ranges concatenated in order). Lets the server grab the
// picked frame from the ORIGINAL movie at full quality.
function clipTimeToSource(t) {
  let acc = 0;
  for (const row of rangesEl.children) {
    const s = parseTs(row.querySelector('.r-start').value);
    const e = parseTs(row.querySelector('.r-end').value);
    if (s == null || e == null || e <= s) continue;
    const d = e - s;
    if (t <= acc + d) return s + (t - acc);
    acc += d;
  }
  return null;
}

async function grabThumbnail() {
  if (!rawClipUrl) return false;
  const btn = document.getElementById('thumb-grab');
  const status = document.getElementById('thumb-status');
  const at = document.getElementById('thumb-video').currentTime || 0;
  const pan = parseFloat(thumbPan.value) / 100;
  btn.disabled = true;
  status.className = 'thumb-status';
  status.textContent = `Rendering thumbnail at ${at.toFixed(1)}s…`;
  // Capture the EXACT pixels on screen: Safari's scrub preview snaps to
  // keyframes, so the frame at `currentTime` can differ from the one being
  // displayed. The canvas capture is what-you-see-is-what-you-get; the
  // timestamp goes along only as a fallback.
  let frame = null;
  try {
    const v = document.getElementById('thumb-video');
    if (v.videoWidth > 0) {
      const c = document.createElement('canvas');
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0);
      frame = c.toDataURL('image/jpeg', 0.92);
    }
  } catch (err) { frame = null; }
  try {
    const res = await fetch(BASE_PATH + '/make-thumbnail', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        video: rawClipUrl,
        at,
        pan,
        frame,
        source_video: videoInput.value.trim() || null,
        source_at: clipTimeToSource(at),
        title: document.getElementById('thumb-title').value.trim(),
        movie: document.getElementById('thumb-movie').value.trim(),
        font: document.getElementById('thumb-font').value,
        color: document.getElementById('thumb-color').value,
        width: shapeInfo().thumb[0],
        height: shapeInfo().thumb[1],
        ...enhancePayload(),
      }),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    const d = await res.json();
    const thumb = document.getElementById('result-thumb');
    thumb.src = d.thumbnail_url;
    thumb.style.display = 'block';
    status.className = 'thumb-status ok';
    status.textContent = `✓ thumbnail set from ${at.toFixed(1)}s`;
    thumbGrabbed = true;
    return true;
  } catch (err) {
    status.textContent = 'Failed: ' + String(err.message).slice(0, 160);
    return false;
  } finally {
    btn.disabled = false;
  }
}
document.getElementById('thumb-grab').addEventListener('click', grabThumbnail);

function buildPayload(ranges, usePrepared) {
  const spanOn = document.getElementById('span_enable') && document.getElementById('span_enable').checked;
  return {
    video: document.getElementById('video').value.trim(),
    ranges,
    source_title: document.getElementById('source_title').value.trim(),
    // Movie span (only when enabled) + hook + speech-snap toggle.
    span_start: spanOn ? parseFloat(document.getElementById('span_start').value) : null,
    curated_region: (typeof curatedRegion !== 'undefined' ? curatedRegion : null),
    span_end: spanOn ? parseFloat(document.getElementById('span_end').value) : null,
    hook: document.getElementById('hook').checked,
    no_snap_speech: !document.getElementById('snap_speech').checked,
    // Only a prepared thumbnail's title is an intentional override. The
    // completed result also fills this field and must not title the next run.
    title: usePrepared ? document.getElementById('thumb-title').value.trim() : '',
    whisper_model: document.getElementById('whisper_model').value,
    ollama_model: document.getElementById('ollama_model').value.trim(),
    editorial_file: document.getElementById('editorial_file').value.trim(),
    watermark: document.getElementById('watermark').value,
    bg_mode: bgMode.value,
    bg_music: bgFileSel.value,
    bg_volume: parseFloat(bgVolume.value),
    bg_skip: parseFloat(document.getElementById('bg_skip').value) || 0,
    size: parseInt(document.getElementById('size').value, 10),
    language: document.getElementById('language').value.trim(),
    // The axes are what the pipeline branches on; `mode` rides along only as the
    // preset label for logging.
    mode: formatSel.value,
    selection: sel(),
    voice: voice(),
    narrate: narrateChk.checked,
    dialogue_volume: parseFloat(dialogueVol.value),
    audio_mix: voice() === 'commentary' ? document.getElementById('audio_mix').value : null,
    narration_volume: parseFloat(nVolume.value),
    end_card: document.getElementById('end_card').checked,
    end_card_duration: parseFloat(document.getElementById('end_card_duration').value),
    tts_voice: document.getElementById('tts_voice').value,
    // Recap-only options (ignored by the server for other modes).
    outro_seconds: parseFloat(outroSec.value),
    no_outro: !recapOutroChk.checked,
    auto_outro_shots: parseInt(document.getElementById('auto_outro_shots').value, 10) || 0,
    outro_shots: document.getElementById('outro_shots_manual').value.trim(),
    target_seconds: parseInt(targetLen.value, 10),
    recap_model: subtitleModel(),
    // Edit-style description credits.
    edit_credits: editCreditsChk.checked,
    music_credit: musicCreditInput.value.trim(),
    editing_program: document.getElementById('editing_program').value.trim(),
    no_disclaimer: !document.getElementById('edit_disclaimer').checked,
    srt: (useSrtChk.checked && probedSrt) ? probedSrt : null,
    no_srt: !useSrtChk.checked,
    use_prepared: !!usePrepared,
    prepared_dir: usePrepared ? preparedDir : null,
    prepared_ranges: usePrepared ? preparedRanges : null,
    remove_silence: document.getElementById('remove_silence').checked,
    no_loudnorm: !document.getElementById('loudnorm').checked,
    mirror: document.getElementById('mirror').checked,
    // shapeKey() already resolves the narrated lock, so the server never has to
    // silently override what the form said.
    aspect: shapeKey(),
    // Two SEPARATE decisions that used to be one checkbox, which is how every
    // render silently stopped producing a thumbnail: the switch above only picks
    // the FRAME (by hand vs automatically). Whether a thumbnail exists at all is
    // its own opt-out, because unticking "pick it by hand" must still leave you
    // with a thumbnail — the note under the switch promises exactly that.
    want_thumbnail: !document.getElementById('skip_thumbnail').checked,
    pick_thumbnail: wantThumbChk.checked,
    caption_font: document.getElementById('caption_font').value,
    caption_font_shuffle: document.getElementById('caption_font_shuffle').checked,
    caption_italic: document.getElementById('caption_italic').checked,
    caption_animation: document.getElementById('caption_animation').value,
    caption_shuffle: document.getElementById('caption_shuffle').checked,
    caption_color: document.getElementById('caption_color').value,
  };
}

// Show the fixed-colour picker only when colour shuffle is off.
(function () {
  const chk = document.getElementById('caption_shuffle');
  const wrap = document.getElementById('caption_color_wrap');
  if (chk && wrap) {
    const sync = () => { wrap.style.display = chk.checked ? 'none' : 'block'; };
    chk.addEventListener('change', sync);
    sync();
  }
})();

// The flow follows the thumbnail switch, not the format: hand-picking a frame
// needs the clip cut first (two phases), and turning it off renders in one.
const MODE_LABEL = {narrated: 'narrated clip', storytelling: 'storytelling recap',
                    recap: 'recap', classic: 'clip', custom: 'edit'};
function submitLabel() {
  const wantThumb = wantThumbChk.checked && !narrationLed();
  submitBtn.textContent = wantThumb
    ? '① Cut clip & pick thumbnail'
    : '🎬 Render ' + (MODE_LABEL[formatSel.value] || 'edit');
  const bits = [];
  if (narrationLed()) {
    bits.push('Writes the script, speaks it, then cuts each clip to its line — one pass.');
  } else if (wantThumb) {
    bits.push('Cuts first so you can scrub to a frame — you render in step ②.');
    if (sel() === 'spine') {
      bits.push('Build the beats above first; there is nothing to cut without them.');
    }
  } else {
    bits.push('Renders in one pass and composes the thumbnail automatically.');
  }
  submitHint.innerHTML = bits.join('<br>');
}
submitLabel();
syncFormat();
probeVideo();

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorBox.style.display = 'none';
  const ranges = collectRanges();
  const wantThumb = wantThumbChk.checked;
  const fail = (msg) => { errorBox.style.display = 'block'; errorBox.textContent = msg; };

  // Every selection agent reads the whole subtitle file.
  if (needsSubs() && !(useSrtChk.checked && probedSrt)) {
    fail('This edit needs a subtitle file — none is selected for this movie.');
    return;
  }
  // Narration-led: the clips do not exist until the script has been spoken, so
  // there is nothing to cut ahead of time and nothing to pick a frame from.
  if (narrationLed()) {
    await doRun('', false);
    return;
  }
  if (sel() === 'spine') {
    // With no ranges, main.py builds the beats itself at render time. That is an
    // LLM call, so its result cannot be cut ahead of the thumbnail picker.
    if (wantThumb && !ranges) {
      fail('Build the beats first (✨ above), or turn off “Pick the thumbnail by hand” to render in one pass.');
      return;
    }
    await (wantThumb ? doPrepare(ranges) : doRun(ranges, false));
    return;
  }
  if (!ranges && sel() !== 'scene') {
    fail('Add at least one valid range.');
    return;
  }
  if (!ranges) {          // scene: the agent picks it at render time
    await doRun('', false);
    return;
  }
  await (wantThumb ? doPrepare(ranges) : doRun(ranges, false));
});

function resetRunUi() {
  // Flush everything a previous run left on screen — its thumbnail, links,
  // log, and step bars must not leak into the next run's thumbnail picker.
  const thumb = document.getElementById('result-thumb');
  thumb.style.display = 'none';
  thumb.removeAttribute('src');
  document.getElementById('result-desc').style.display = 'none';
  document.getElementById('result-meta').style.display = 'none';
  document.getElementById('result-narration').style.display = 'none';
  document.getElementById('result-video').style.display = 'none';
  document.getElementById('thumb-status').className = 'thumb-status';
  document.getElementById('thumb-status').textContent = '';
  logEl.textContent = '';
  setStep(0);
  stepLabel.className = '';
  stepLabel.textContent = '';
  progressEl.style.display = 'none';
}

async function doPrepare(ranges) {
  resultEl.style.display = 'none';
  document.getElementById('thumb-editor').style.display = 'none';
  resetRunUi();
  rawClipUrl = null;
  preparedDir = null;
  preparedRanges = null;
  thumbGrabbed = false;
  submitBtn.disabled = true;
  submitBtn.textContent = 'Cutting clip…';
  try {
    const res = await fetch(BASE_PATH + '/prepare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        video: document.getElementById('video').value.trim(),
        ranges,
        source_title: document.getElementById('source_title').value.trim(),
        // Phase 1 cuts on the music's beats when the music leads, so it needs
        // the same axes the render will use.
        mode: formatSel.value,
        selection: sel(),
        voice: voice(),
        srt: (useSrtChk.checked && probedSrt) ? probedSrt : null,
        bg_mode: bgMode.value,
        bg_music: bgFileSel.value,
      }),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    const d = await res.json();
    // Reveal the thumbnail picker loaded with the freshly-cut raw clip.
    rawClipUrl = d.raw_clip_url;
    preparedDir = d.clip_dir || null;
    preparedRanges = d.prepared_ranges || ranges;
    // A "random" recap track is rolled server-side in phase 1. Pin it, or phase 2
    // would roll a different one, snap to different beats, and drift the captions
    // off the very cut this thumbnail was picked from.
    if (d.recap_music) {
      bgMode.value = 'file';
      bgFileSel.value = d.recap_music;
      syncBgMode();
      syncMusicNote();
    }
    resultEl.style.display = 'block';
    document.getElementById('result-title').textContent = 'Pick your thumbnail, then render';
    thumbVideo.src = rawClipUrl;
    document.getElementById('thumb-title').value = '';
    titleOptionsEl.innerHTML = '';   // options belong to the previous cut
    document.getElementById('thumb-movie').value = d.movie || '';
    document.getElementById('thumb-editor').style.display = 'block';
    document.getElementById('thumb-render').style.display = 'block';
    document.getElementById('thumb-status').textContent = 'Scrub to a frame, set the crop, grab it — then render.';
  } catch (err) {
    errorBox.style.display = 'block';
    errorBox.textContent = 'Prepare failed: ' + err.message;
  } finally {
    submitBtn.disabled = false;
    submitLabel();
  }
}

async function doRun(ranges, usePrepared, overrides = {}) {
  resultEl.style.display = usePrepared ? 'block' : 'none';
  logEl.textContent = '';
  setStep(0);
  stepLabel.className = '';
  stepLabel.textContent = 'Starting…';
  wsHandledDone = false;
  const payload = Object.assign(buildPayload(ranges, usePrepared), overrides);
  submitBtn.disabled = true;
  submitBtn.textContent = 'Running…';
  const renderBtn = document.getElementById('thumb-render');
  renderBtn.disabled = true;
  progressEl.style.display = 'block';
  try {
    const res = await fetch(BASE_PATH + '/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    // Draw the correct number of segments up front — waiting for the first
    // '[n/N]' log line meant the bar showed a stale guess until then.
    const d = await res.json();
    buildSteps(d.total_steps);
  } catch (err) {
    submitBtn.disabled = false;
    submitLabel();
    renderBtn.disabled = false;
    errorBox.style.display = 'block';
    errorBox.textContent = 'Failed to start: ' + err.message;
  }
}

document.getElementById('thumb-render').addEventListener('click', async () => {
  const ranges = collectRanges();
  if (!ranges) return;
  // Forgot (or skipped) the grab? Bake whatever frame is currently scrubbed —
  // rendering with an auto-generated thumbnail the user never saw is exactly
  // the "thumbnail wasn't the one I picked" surprise.
  if (!thumbGrabbed && rawClipUrl) {
    await grabThumbnail();
  }
  doRun(ranges, true);
});

// ── Style persistence + title reset (user requests) ─────────────────────────
// Font/style dropdowns REMEMBER the previous selection across renders; the
// previous render's TITLE is cleared when you switch movie or format (it was
// sticking around until a server restart).
(function () {
  try {
    for (const id of ['caption_font', 'thumb-font', 'thumb-color', 'thumb-enhance']) {
      const el = document.getElementById(id);
      if (!el) continue;
      const key = 'clip_pref_' + id;
      const saved = localStorage.getItem(key);
      if (saved !== null && [...el.options].some(o => o.value === saved)) el.value = saved;
      el.addEventListener('change', () => { try { localStorage.setItem(key, el.value); } catch (e) {} });
    }
  } catch (e) {}

  // Effect segmented button group → hidden #caption_animation (persisted).
  try {
    const grp = document.getElementById('caption_animation_btns');
    const hidden = document.getElementById('caption_animation');
    if (grp && hidden) {
      const key = 'clip_pref_caption_animation';
      const set = (val) => {
        hidden.value = val;
        for (const b of grp.querySelectorAll('.seg')) b.classList.toggle('active', b.dataset.value === val);
        try { localStorage.setItem(key, val); } catch (e) {}
      };
      const saved = localStorage.getItem(key);
      if (saved && grp.querySelector('.seg[data-value="' + saved + '"]')) set(saved);
      grp.addEventListener('click', (e) => { const b = e.target.closest('.seg'); if (b) set(b.dataset.value); });
    }
  } catch (e) {}
  const clearTitle = () => {
    const t = document.getElementById('thumb-title'); if (t) t.value = '';
    const o = document.getElementById('title-options'); if (o) o.innerHTML = '';
  };
  const vid = document.getElementById('video'); if (vid) vid.addEventListener('change', clearTitle);
  const fc = document.getElementById('fmt-cards'); if (fc) fc.addEventListener('click', clearTitle);
})();

// ── Movie SPAN slider + hook/snap wiring ────────────────────────────────────
let movieDuration = 0;
function _fmtT(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), ss = s % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
           : `${m}:${String(ss).padStart(2, '0')}`;
}
function syncSpan() {
  const a = document.getElementById('span_start'), b = document.getElementById('span_end');
  const fv = document.getElementById('span-from-val'), tv = document.getElementById('span-to-val');
  if (!a || !b) return;
  if (+a.value >= +b.value - 5) a.value = Math.max(0, +b.value - 5);
  if (fv) fv.textContent = _fmtT(+a.value);
  if (tv) tv.textContent = (movieDuration && +b.value >= movieDuration) ? 'end' : _fmtT(+b.value);
}
function setSpanBounds(dur) {
  movieDuration = dur;
  const a = document.getElementById('span_start'), b = document.getElementById('span_end');
  if (!a || !b) return;
  a.max = Math.floor(dur); b.max = Math.floor(dur);
  if (+b.value === 0 || +b.value > dur) b.value = Math.floor(dur);
  syncSpan();
}
(function () {
  const en = document.getElementById('span_enable'), wrap = document.getElementById('span-wrap');
  if (en && wrap) en.addEventListener('change', () => { wrap.style.display = en.checked ? 'block' : 'none'; });
  for (const id of ['span_start', 'span_end']) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', syncSpan);
  }
  syncSpan();
})();
let comparisonPlan = null;
const comparisonReview = document.getElementById('comparison-review');
const comparisonApproved = document.getElementById('comparison-approved');
const comparisonBuild = document.getElementById('comparison-build');
function comparisonButtons() {
  for (const style of ['short', 'long']) document.getElementById('comparison-' + style).disabled = !comparisonPlan || !comparisonApproved.checked;
}
for (const id of ['video', 'source_title', 'span_start', 'span_end', 'span_enable', 'comparison-duration', 'ollama_model', 'comparison-feedback']) {
  document.getElementById(id).addEventListener('input', () => {
    comparisonPlan = null; comparisonApproved.checked = false; comparisonButtons();
    comparisonReview.textContent = 'Settings changed — build a fresh comparison plan.';
  });
}
comparisonApproved.addEventListener('change', comparisonButtons);
comparisonBuild.addEventListener('click', async () => {
  if (submitBtn.disabled) return;
  comparisonPlan = null; comparisonApproved.checked = false; comparisonButtons();
  comparisonBuild.disabled = true;
  comparisonReview.textContent = 'Establishing characters, planning the story, checking facts and extracting preview frames…';
  try {
    if (!document.getElementById('span_enable').checked) throw new Error('Enable a source window first.');
    const payload = buildPayload('', false);
    payload.target_seconds = Number(document.getElementById('comparison-duration').value);
    payload.editorial_feedback = document.getElementById('comparison-feedback').value;
    const res = await fetch(BASE_PATH + '/comparison-plan', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if (!res.ok) throw new Error(await res.text());
    const job = await res.json();
    let result;
    do {
      await new Promise(resolve => setTimeout(resolve, 1500));
      const status = await fetch(BASE_PATH + '/comparison-plan/' + job.job_id);
      if (!status.ok) throw new Error(await status.text());
      result = await status.json();
    } while (result.status === 'running');
    if (result.status !== 'complete') throw new Error(result.error);
    const plan = result.plan;
    comparisonPlan = plan;
    comparisonReview.textContent = plan.title + '\\n' + plan.duration + ' seconds each. Structure checks passed; inspect the attribution review below. Visual and rights review remain separate.\\n' + plan.story + '\\n\\n';
    const notes = document.createElement('p'); notes.textContent = 'Attribution review: ' + (plan.review_notes.length ? plan.review_notes.join(' / ') : 'No model issues reported; visual confirmation still required.'); comparisonReview.appendChild(notes);
    for (const c of plan.characters) {
      const text = document.createElement('p');
      text.textContent = c.name + ' — ' + c.role + ' (' + c.confidence + ') Evidence: ' + c.evidence;
      comparisonReview.appendChild(text);
    }
    for (const b of plan.beats) {
      const text = document.createElement('p');
      text.textContent = b.role + '\\nNarration: ' + b.narration + '\\nByline: ' + b.byline + '\\nLong source: ' + b.long_start + '; short sources: ' + b.short_starts.join(', ') + '\\nEvidence: ' + b.evidence;
      comparisonReview.appendChild(text);
      for (const url of b.previews) {
        const img = document.createElement('img'); img.src = url; img.alt = b.role + ' candidate frame'; img.style.width = '30%';
        comparisonReview.appendChild(img);
      }
    }
  } catch (err) { comparisonPlan = null; comparisonReview.textContent = 'Plan held: ' + err.message; }
  finally { comparisonBuild.disabled = false; comparisonButtons(); }
});
for (const style of ['short', 'long']) document.getElementById('comparison-' + style).addEventListener('click', async () => {
  if (!comparisonPlan || !comparisonApproved.checked || submitBtn.disabled) return;
  const plan = comparisonPlan;
  await doRun(plan[style].ranges, false, {
    video:plan.video, source_title:document.getElementById('source_title').value.trim(),
    selection:'manual', voice:'commentary', aspect:style === 'short' ? 'vertical' : 'landscape',
    editorial_file:plan[style].editorial_file, comparison_style:style, visual_reviewed:true,
    title:'', hook:false, no_snap_speech:true, remove_silence:false, use_prepared:false,
    audio_mix:style === 'short' ? 'original' : 'narration-only', narration_volume:1,
    dialogue_volume:style === 'short' ? 1 : 0, bg_mode:'none', edit_credits:false, want_thumbnail:true,
  });
});
</script>
</body>
</html>
"""


_FAVICON_SVG = """\
<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <clipPath id="c"><circle cx="32" cy="32" r="32"/></clipPath>
  </defs>
  <image href="__BASE_PATH__/favicon.png" x="0" y="0" width="64" height="64"
         clip-path="url(#c)" preserveAspectRatio="xMidYMid meet"/>
</svg>
"""


def _render_html() -> str:
    return _HTML.replace("__BASE_PATH__", BASE_PATH)


def _render_favicon_svg() -> str:
    return _FAVICON_SVG.replace("__BASE_PATH__", BASE_PATH)


if __name__ == "__main__":
    main()
