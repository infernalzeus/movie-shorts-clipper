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
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

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
THUMBNAIL_SCRIPT = SRC_DIR / "thumbnail.py"


def _clean_path(value: str | None) -> str:
    """Strip whitespace and any surrounding quotes from a pasted file path.

    Windows' 'Copy as path' wraps the path in double quotes, and users often
    paste those in — accept the path anyway instead of reporting 'not found'.
    """
    s = (value or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("\"", "'"):
        s = s[1:-1]
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
        self.total_steps: int = 6
        self.done: bool = False
        self.ok: bool = False
        self.result: dict = {}
        self.subscribers: set[web.WebSocketResponse] = set()

    def reset(self) -> None:
        self.proc = None
        self.lines = []
        self.step = 0
        self.total_steps = 6
        self.done = False
        self.ok = False
        self.result = {}


run_state = RunState()

# Phase-1 "prepared" clip (cut, awaiting a thumbnail pick before the full render).
_prepared: dict = {"clip_dir": None, "raw_clip_url": None}
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

            await _broadcast({"type": "line", "text": line, "step": run_state.step, "total": run_state.total_steps})

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
    return web.json_response({"exists": True, "srt": str(srt) if srt else None})


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

    if _suggest_lock.locked():
        raise web.HTTPConflict(text="A scene suggestion is already running")
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


_thumb_lock = asyncio.Lock()


@routes.post("/make-thumbnail")
async def make_thumbnail_route(request: web.Request) -> web.Response:
    """Regenerate the thumbnail from a user-picked frame of the raw clip.

    Body: {video: "<served /output url of the raw clip>", at: <seconds>,
           title: "...", movie: "..."}. Writes thumbnail.jpg beside the clip.
    """
    data = await request.json()
    src = _resolve_output(data.get("video") or "")
    if src is None or not src.is_file():
        raise web.HTTPBadRequest(text="Source clip not found")

    out_path = src.parent / "thumbnail.jpg"
    args = [
        sys.executable, str(THUMBNAIL_SCRIPT),
        "--video", str(src),
        "--out", str(out_path),
    ]
    at = data.get("at")
    if at is not None:
        try:
            args += ["--at", f"{float(at):.3f}"]
        except (TypeError, ValueError):
            raise web.HTTPBadRequest(text="Invalid timestamp")
    pan = data.get("pan")
    if pan is not None:
        try:
            args += ["--pan", f"{min(1.0, max(0.0, float(pan))):.4f}"]
        except (TypeError, ValueError):
            raise web.HTTPBadRequest(text="Invalid pan")
    if data.get("title"):
        args += ["--title", str(data["title"])]
    if data.get("movie"):
        args += ["--movie", str(data["movie"])]

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

    if _prepare_lock.locked():
        raise web.HTTPConflict(text="Already preparing a clip")
    async with _prepare_lock:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=str(SRC_DIR), env=env,
        )
        stdout, _ = await proc.communicate()

    out = stdout.decode(errors="replace")
    if proc.returncode != 0:
        raise web.HTTPInternalServerError(text=f"prepare failed:\n{out[-800:]}")

    clip_dir = raw_clip = movie = None
    for line in out.splitlines():
        if "Clip dir:" in line:
            clip_dir = line.split("Clip dir:")[-1].strip()
        elif "Raw clip:" in line:
            raw_clip = line.split("Raw clip:")[-1].strip()
        elif "Movie:" in line:
            movie = line.split("Movie:")[-1].strip()
    if not clip_dir or not raw_clip:
        raise web.HTTPInternalServerError(text="prepare produced no clip")

    _prepared["clip_dir"] = clip_dir
    _prepared["raw_clip_url"] = _rel_output_url(raw_clip)
    return web.json_response({
        "ok": True,
        "raw_clip_url": _prepared["raw_clip_url"],
        "movie": movie or "",
    })


@routes.post("/run")
async def start_run(request: web.Request) -> web.Response:
    if run_state.proc is not None and run_state.proc.returncode is None:
        raise web.HTTPConflict(text="A clip is already running")

    data = await request.json()

    video = _clean_path(data.get("video"))
    ranges = (data.get("ranges") or "").strip()
    if not video or not Path(video).is_file():
        raise web.HTTPBadRequest(text="Video file not found")
    if not ranges:
        raise web.HTTPBadRequest(text="At least one range is required")

    args = ["--video", video, "--ranges", ranges]

    if data.get("source_title"):
        args += ["--source-title", str(data["source_title"])]
    args += ["--whisper-model", data.get("whisper_model") or "medium"]
    args += ["--ollama-model", data.get("ollama_model") or "nemotron-3-super:cloud"]
    args += ["--language", data.get("language") or "en"]
    if data.get("size"):
        args += ["--size", str(data["size"])]

    watermark = data.get("watermark") or "mv-edits"
    args += ["--watermark", watermark]

    if data.get("narrate"):
        args += ["--narrate"]
        if data.get("narration_volume") is not None:
            args += ["--narration-volume", str(data["narration_volume"])]
        if data.get("tts_voice"):
            args += ["--tts-voice", str(data["tts_voice"])]
        # Two-phase: reuse the prepared cut + hand-picked thumbnail, and bake the
        # thumbnail into the first 0.5s so it can be chosen on YouTube mobile.
        if data.get("use_prepared") and _prepared.get("clip_dir"):
            args += ["--clip-dir", str(_prepared["clip_dir"]), "--prepend-thumbnail"]
    if data.get("no_srt"):
        args += ["--no-srt"]
    elif data.get("srt") and Path(str(data["srt"])).is_file():
        args += ["--srt", str(data["srt"])]
    if data.get("layout") in ("square", "vertical"):
        args += ["--layout", data["layout"]]

    bg_mode = data.get("bg_mode") or "random"
    if bg_mode == "none":
        args += ["--no-bg-music"]
    elif bg_mode == "file" and data.get("bg_music"):
        bg_path = AUDIO_DIR / str(data["bg_music"])
        if bg_path.is_file():
            args += ["--bg-music", str(bg_path)]
    if data.get("bg_volume") is not None:
        args += ["--bg-volume", str(data["bg_volume"])]

    run_state.reset()
    asyncio.create_task(_stream_run(args))

    return web.json_response({"ok": True})


@routes.get("/ws")
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    run_state.subscribers.add(ws)

    for line in run_state.lines:
        await ws.send_str(json.dumps({
            "type": "line", "text": line,
            "step": run_state.step, "total": run_state.total_steps,
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


async def _terminate_run(app: web.Application) -> None:
    """On shutdown, kill any in-flight render and its whole child tree so a
    stopped/restarted server (e.g. Agent Hub cycling the app) never leaves
    orphaned ffmpeg/edge-tts processes behind."""
    proc = run_state.proc
    if proc is None or proc.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            # terminate() only signals main.py; taskkill /T reaps the ffmpeg tree.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)
    except (ProcessLookupError, asyncio.TimeoutError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    except Exception:
        pass


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
    app = web.Application()
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
</style>
</head>
<body>

<h1>🎬 Movie Clipper</h1>
<div class="sub">movie-shorts-clipper — cut, caption, and upload-ready in one run</div>

<form id="form">
  <fieldset>
    <legend>Source</legend>
    <label for="video">Movie file path</label>
    <div class="range-row">
      <input type="text" id="video" placeholder="N:\\Movies\\film.mkv" required>
      <button type="button" id="check-srt" class="mini-btn">Check subtitles</button>
    </div>
    <div id="srt-status"></div>
    <label id="use-srt-wrap" style="display:none;align-items:center;gap:8px;cursor:pointer;margin-top:6px">
      <input type="checkbox" id="use_srt" checked style="accent-color:var(--accent)">
      <span>Use this subtitle file for captions (local, no upload)</span>
    </label>

    <label>Clip range(s)</label>
    <div id="ranges"></div>
    <button type="button" id="add-range" class="mini-btn">+ Add range</button>
    <div id="suggest-wrap" style="display:none;margin-top:10px">
      <label for="suggest_model">Scene suggestion — model (the whole subtitle file is fed to it)</label>
      <select id="suggest_model">
        <option value="gemma4:e4b" selected>gemma4:e4b — local, no upload</option>
        <option value="gemma4:e2b">gemma4:e2b — local, no upload</option>
        <option value="__cloud__">Cloud — Ollama model below (uploads subtitles, uses bandwidth)</option>
      </select>
      <button type="button" id="suggest-btn" class="mini-btn">✨ Suggest scenes from subtitles</button>
    </div>
    <div id="suggestions"></div>

    <label for="source_title">Source title override (optional)</label>
    <input type="text" id="source_title" placeholder="Auto-parsed from filename if left blank">
  </fieldset>

  <fieldset>
    <legend>Narration</legend>
    <label style="display:flex;align-items:center;gap:8px;margin:0;cursor:pointer">
      <input type="checkbox" id="narrate" checked style="accent-color:var(--accent)">
      <span>Narrated vertical format — TTS voiceover + text beneath the video</span>
    </label>
    <div id="narrate-opts">
      <label for="tts_voice">Narrator voice</label>
      <select id="tts_voice">
        <option value="en-US-ChristopherNeural" selected>Christopher (US, deep)</option>
        <option value="en-US-GuyNeural">Guy (US)</option>
        <option value="en-US-JennyNeural">Jenny (US)</option>
        <option value="en-GB-RyanNeural">Ryan (British)</option>
        <option value="en-GB-SoniaNeural">Sonia (British)</option>
        <option value="en-AU-WilliamNeural">William (Australian)</option>
      </select>
      <label for="narration_volume">Voiceover volume <span id="nvolume-val">0.05</span></label>
      <input type="range" id="narration_volume" min="0" max="1" step="0.01" value="0.05">
    </div>
  </fieldset>

  <fieldset>
    <legend>Generation</legend>
    <div class="row">
      <div>
        <label for="whisper_model">Whisper model (fallback, no SRT)</label>
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
    <label for="ollama_model">Ollama model</label>
    <input type="text" id="ollama_model" value="nemotron-3-super:cloud">
  </fieldset>

  <fieldset>
    <legend>Background music</legend>
    <label for="bg_mode">Mode</label>
    <select id="bg_mode">
      <option value="random" selected>Random from audio/</option>
      <option value="file">Specific file</option>
      <option value="none">None</option>
    </select>
    <div id="bg-file-wrap" style="display:none">
      <label for="bg_music">File</label>
      <select id="bg_music"></select>
    </div>
    <label for="bg_volume">Volume <span id="volume-val">0.10</span></label>
    <input type="range" id="bg_volume" min="0" max="1" step="0.01" value="0.10">
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
  </details>

  <button type="submit" id="submit">Generate Clip</button>
</form>

<div id="error"></div>

<div id="progress">
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
    <div class="thumb-hint">Scrub to a frame, then move the crop slider to choose which 9:16 slice of the shot to keep (the highlighted box shows it). Title and movie name are drawn on automatically.</div>
    <div id="thumb-video-wrap">
      <video id="thumb-video" controls preload="metadata"></video>
      <div id="thumb-cropbox"></div>
    </div>
    <label for="thumb-pan">Crop position <span id="thumb-pan-val">center</span></label>
    <input type="range" id="thumb-pan" min="0" max="100" value="50">
    <div class="row">
      <div><label for="thumb-title">Title (top)</label><input type="text" id="thumb-title"></div>
      <div><label for="thumb-movie">Movie name (bottom)</label><input type="text" id="thumb-movie"></div>
    </div>
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
const suggestBtn = document.getElementById('suggest-btn');
const suggestWrap = document.getElementById('suggest-wrap');
const suggestModel = document.getElementById('suggest_model');
const useSrtWrap = document.getElementById('use-srt-wrap');
const useSrtChk  = document.getElementById('use_srt');
const suggestionsEl = document.getElementById('suggestions');
const stepsEl    = document.getElementById('steps');
const narrateChk = document.getElementById('narrate');
const narrateOpts = document.getElementById('narrate-opts');
const nVolume    = document.getElementById('narration_volume');
const nVolumeVal = document.getElementById('nvolume-val');

let probedSrt = null;
let totalSteps = 6;

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

bgMode.addEventListener('change', () => {
  bgFileWrap.style.display = bgMode.value === 'file' ? 'block' : 'none';
});
bgVolume.addEventListener('input', () => { volumeVal.textContent = bgVolume.value; });
nVolume.addEventListener('input', () => { nVolumeVal.textContent = nVolume.value; });
narrateChk.addEventListener('change', () => {
  narrateOpts.style.display = narrateChk.checked ? 'block' : 'none';
});

// ── SRT probe + scene suggestions ──────────────────────────────────────────

async function probeVideo() {
  probedSrt = null;
  suggestWrap.style.display = 'none';
  useSrtWrap.style.display = 'none';
  suggestionsEl.innerHTML = '';
  srtStatus.className = '';
  const video = videoInput.value.trim();
  if (!video) { srtStatus.textContent = 'Enter a movie file path first.'; return; }
  srtStatus.textContent = 'Checking for a subtitle file…';
  checkSrtBtn.disabled = true;
  try {
    const res = await fetch(BASE_PATH + '/probe-video', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({video}),
    });
    const d = await res.json();
    if (!d.exists) {
      srtStatus.textContent = '⚠ file not found at that path';
      return;
    }
    if (d.srt) {
      probedSrt = d.srt;
      srtStatus.className = 'found';
      srtStatus.textContent = '✓ subtitles found: ' + d.srt.split(/[\\\\/]/).pop();
      useSrtWrap.style.display = 'flex';
      suggestWrap.style.display = 'block';
    } else {
      srtStatus.textContent = 'no .srt found next to the movie — Whisper will transcribe';
    }
  } catch (err) {
    srtStatus.textContent = 'Could not check (' + String(err.message).slice(0, 120) + ')';
  } finally {
    checkSrtBtn.disabled = false;
  }
}
checkSrtBtn.addEventListener('click', probeVideo);
// Re-checking whenever the path changes keeps stale results from lingering.
videoInput.addEventListener('change', () => { probedSrt = null; suggestWrap.style.display = 'none';
  useSrtWrap.style.display = 'none'; suggestionsEl.innerHTML = ''; srtStatus.className = '';
  srtStatus.textContent = 'Click “Check subtitles” to look for a matching .srt.'; });

suggestBtn.addEventListener('click', async () => {
  suggestBtn.disabled = true;
  suggestBtn.textContent = '⏳ Analyzing subtitles… (can take a minute)';
  suggestionsEl.innerHTML = '';
  try {
    const res = await fetch(BASE_PATH + '/suggest-scenes', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        srt: probedSrt,
        title: document.getElementById('source_title').value.trim(),
        model: suggestModel.value === '__cloud__'
          ? document.getElementById('ollama_model').value.trim()
          : suggestModel.value,
        count: 5,
      }),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    const d = await res.json();
    renderSuggestions(d.scenes || []);
  } catch (err) {
    suggestionsEl.innerHTML = '<div class="suggestion" style="cursor:default;color:var(--err)">Suggestion failed: '
      + String(err.message).slice(0, 300) + '</div>';
  } finally {
    suggestBtn.disabled = false;
    suggestBtn.textContent = '✨ Suggest scenes from subtitles';
  }
});

function renderSuggestions(scenes) {
  suggestionsEl.innerHTML = '';
  if (!scenes.length) {
    suggestionsEl.innerHTML = '<div class="suggestion" style="cursor:default">No scenes suggested.</div>';
    return;
  }
  for (const sc of scenes) {
    const card = document.createElement('div');
    card.className = 'suggestion';
    card.innerHTML = `
      <div class="s-head"><span>${sc.title || 'Scene'}</span><span class="s-range">${sc.range}</span></div>
      <div class="s-reason">${sc.reason || ''}</div>
    `;
    card.title = 'Click to add as a clip range';
    card.onclick = () => {
      const [s, e] = sc.range.split('-');
      // Reuse the first empty range row before adding a new one
      const empty = [...rangesEl.children].find(r =>
        !r.querySelector('.r-start').value.trim() && !r.querySelector('.r-end').value.trim());
      if (empty) {
        empty.querySelector('.r-start').value = s;
        empty.querySelector('.r-end').value = e;
      } else {
        addRangeRow(s, e);
      }
      card.classList.add('added');
    };
    suggestionsEl.appendChild(card);
  }
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
  totalSteps = total;
  stepsEl.innerHTML = '';
  for (let i = 1; i <= total; i++) {
    const div = document.createElement('div');
    div.className = 'step';
    div.dataset.n = i;
    stepsEl.appendChild(div);
  }
}
buildSteps(6);

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
      if (d.total && d.total !== totalSteps) buildSteps(d.total);
      if (d.step) setStep(d.step);
    } else if (d.type === 'done') {
      wsHandledDone = true;
      submitBtn.disabled = false;
      submitLabel();
      const renderBtn = document.getElementById('thumb-render');
      renderBtn.disabled = false;
      if (d.ok) {
        renderBtn.style.display = 'none';   // phase-2 done — thumbnail already baked in
        setStep(totalSteps + 1);
        showResult(d.result);
      } else {
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
function showResult(result) {
  resultEl.style.display = 'block';
  document.getElementById('result-title').textContent = result.title || 'Clip ready';
  const video = document.getElementById('result-video');
  if (result.video_url) { video.src = result.video_url; video.style.display = 'block'; }
  const descA = document.getElementById('result-desc');
  const metaA = document.getElementById('result-meta');
  if (result.description_url) descA.href = result.description_url; else descA.style.display = 'none';
  if (result.metadata_url) metaA.href = result.metadata_url; else metaA.style.display = 'none';
  const thumb = document.getElementById('result-thumb');
  if (result.thumbnail_url) { thumb.src = result.thumbnail_url; thumb.style.display = 'block'; }
  const narrA = document.getElementById('result-narration');
  if (result.narration_url) { narrA.href = result.narration_url; narrA.style.display = 'inline-block'; }

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
let cropWidthFrac = 1;   // width of the 9:16 window as a fraction of the frame width

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
  const aspect = thumbVideo.videoWidth / thumbVideo.videoHeight;
  // A 9:16 window is (9/16)/aspect of the frame's width.
  cropWidthFrac = Math.min(1, (9 / 16) / aspect);
  updateCropbox();
});

document.getElementById('thumb-grab').addEventListener('click', async () => {
  if (!rawClipUrl) return;
  const btn = document.getElementById('thumb-grab');
  const status = document.getElementById('thumb-status');
  const at = document.getElementById('thumb-video').currentTime || 0;
  const pan = parseFloat(thumbPan.value) / 100;
  btn.disabled = true;
  status.className = 'thumb-status';
  status.textContent = `Rendering thumbnail at ${at.toFixed(1)}s…`;
  try {
    const res = await fetch(BASE_PATH + '/make-thumbnail', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        video: rawClipUrl,
        at,
        pan,
        title: document.getElementById('thumb-title').value.trim(),
        movie: document.getElementById('thumb-movie').value.trim(),
      }),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    const d = await res.json();
    const thumb = document.getElementById('result-thumb');
    thumb.src = d.thumbnail_url;
    thumb.style.display = 'block';
    status.className = 'thumb-status ok';
    status.textContent = `✓ thumbnail set from ${at.toFixed(1)}s`;
  } catch (err) {
    status.textContent = 'Failed: ' + String(err.message).slice(0, 160);
  } finally {
    btn.disabled = false;
  }
});

function buildPayload(ranges, usePrepared) {
  return {
    video: document.getElementById('video').value.trim(),
    ranges,
    source_title: document.getElementById('source_title').value.trim(),
    whisper_model: document.getElementById('whisper_model').value,
    ollama_model: document.getElementById('ollama_model').value.trim(),
    watermark: document.getElementById('watermark').value,
    bg_mode: bgMode.value,
    bg_music: bgFileSel.value,
    bg_volume: parseFloat(bgVolume.value),
    size: parseInt(document.getElementById('size').value, 10),
    language: document.getElementById('language').value.trim(),
    narrate: narrateChk.checked,
    narration_volume: parseFloat(nVolume.value),
    tts_voice: document.getElementById('tts_voice').value,
    srt: (useSrtChk.checked && probedSrt) ? probedSrt : null,
    no_srt: !useSrtChk.checked,
    use_prepared: !!usePrepared,
  };
}

// Narrated runs are two-phase: cut first so you can pick a thumbnail, then render.
function submitLabel() {
  submitBtn.textContent = narrateChk.checked ? '① Cut clip & pick thumbnail' : 'Generate Clip';
}
narrateChk.addEventListener('change', submitLabel);
submitLabel();

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorBox.style.display = 'none';
  const ranges = collectRanges();
  if (!ranges) {
    errorBox.style.display = 'block';
    errorBox.textContent = 'Add at least one valid range.';
    return;
  }
  if (narrateChk.checked) {
    await doPrepare(ranges);
  } else {
    await doRun(ranges, false);
  }
});

async function doPrepare(ranges) {
  resultEl.style.display = 'none';
  document.getElementById('thumb-editor').style.display = 'none';
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
      }),
    });
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    const d = await res.json();
    // Reveal the thumbnail picker loaded with the freshly-cut raw clip.
    rawClipUrl = d.raw_clip_url;
    resultEl.style.display = 'block';
    document.getElementById('result-title').textContent = 'Pick your thumbnail, then render';
    document.getElementById('result-video').style.display = 'none';
    thumbVideo.src = rawClipUrl;
    document.getElementById('thumb-title').value = '';
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

async function doRun(ranges, usePrepared) {
  resultEl.style.display = usePrepared ? 'block' : 'none';
  logEl.textContent = '';
  setStep(0);
  wsHandledDone = false;
  const payload = buildPayload(ranges, usePrepared);
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
  } catch (err) {
    submitBtn.disabled = false;
    submitLabel();
    renderBtn.disabled = false;
    errorBox.style.display = 'block';
    errorBox.textContent = 'Failed to start: ' + err.message;
  }
}

document.getElementById('thumb-render').addEventListener('click', () => {
  const ranges = collectRanges();
  if (ranges) doRun(ranges, true);
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
