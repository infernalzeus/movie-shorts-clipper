# movie-shorts-clipper

Turn any movie or TV file into a upload-ready YouTube Short. Give it a file and one or more timestamp ranges — it cuts the clip, formats it for vertical, captions it, and generates a YouTube title/description/tags via a local Ollama model.

Two output formats:

- **Classic** — square center crop with word-pop karaoke captions (the original pipeline).
- **Narrated** (`--narrate`) — the monetization-friendly format: the square crop sits on a black 9:16 canvas, an LLM writes a scene narration that an edge-tts voice reads at low volume (~5%), the narration text appears beneath the video in sync, dialogue captions come from the movie's own `.srt` file, and a thumbnail is extracted from the clip. The original narration layer makes the content transformative rather than a raw re-upload.

## What you need installed first

- **ffmpeg** on PATH, compiled with `libass` — verify with `ffmpeg -filters | grep ass`
- **Ollama** running at `http://localhost:11434` with a model pulled. Default: `nemotron-3-super:cloud`.
  - Avoid reasoning models like `qwen3.6` — they return output in a `thinking` field under `format=json`, which breaks metadata parsing.
  - Cloud models (`*:cloud`) send prompts over the internet. Scene suggestion feeds a **whole movie's subtitles** into the prompt — use a local model (e.g. `gemma4:e4b`) for that unless you're happy uploading it.
- **Python 3.11+** and dependencies: `pip install -r requirements.txt`
- **faster-whisper** will download the Whisper model on first run (~1.5 GB for `medium`). Only needed when no `.srt` is available.
- **edge-tts** (installed via requirements) needs internet access while synthesizing the narration voiceover — it streams a few hundred KB per clip from Microsoft's TTS service.

## Usage

```
cd src
python main.py
```

Prompts for the movie file path and clip range(s). Or skip the prompts:

```
python main.py \
  --video "C:\movies\The.Last.Heist.2019.1080p.BluRay.x264.mkv" \
  --ranges "3:20-3:30; 5:15-5:45"
```

Multiple ranges are concatenated into a single clip. Ranges are semicolon or comma separated.

Narrated format:

```
python main.py \
  --video "N:\Movies\Legend.2015.1080p.mp4" \
  --ranges "1:58-2:28" \
  --narrate --ollama-model gemma4:e4b
```

If a `.srt` sits next to the movie it is auto-detected and used for the dialogue captions (no Whisper run); `--no-srt` forces Whisper instead.

### Scene suggestions

Don't know which scene to clip? Let an LLM read the subtitles and propose shorts-worthy moments:

```
python scenes.py --srt "N:\Movies\English.srt" --title "Legend" --model gemma4:e4b --count 5
```

Prints ranges you can paste straight into `--ranges` (add `--json` for machine-readable output). The whole subtitle file goes into the prompt, so prefer a local model.

## All flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--video` | (prompt) | Path to source movie file |
| `--ranges` | (prompt) | Clip ranges, e.g. `3:20-3:30; 5:15-5:45` |
| `--source-title` | parsed from filename | Movie/show name for metadata — override if auto-parse looks off |
| `--narrate` | off | Narrated vertical format: LLM narration + TTS + text band + thumbnail |
| `--narration-volume` | `0.05` | Narration voiceover volume (0.0–1.0) |
| `--tts-voice` | `en-US-ChristopherNeural` | edge-tts voice for the narration |
| `--srt` | auto-detected | Subtitle file for dialogue captions |
| `--no-srt` | — | Ignore subtitle files, transcribe with Whisper |
| `--layout` | `vertical` if narrating, else `square` | `square` or `vertical` (square-on-black 9:16) |
| `--prepend-thumbnail` | off | Bake the thumbnail as the first 0.5s of the final video (pick it as the Short's thumbnail on mobile) |
| `--clip-dir` | (auto) | Use/reuse an exact output dir — keeps its `clip_raw.mp4` and hand-picked `thumbnail.jpg` |
| `--prepare-only` | — | Only cut the raw clip into `--clip-dir`, then exit (phase 1 of the two-phase thumbnail flow) |
| `--whisper-model` | `medium` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--ollama-model` | `nemotron-3-super:cloud` | Ollama model for narration + title/description/tags |
| `--size` | `1080` | Square side length in pixels |
| `--bg-music` | random from `audio/` (off in narrated mode) | Path to a background music file |
| `--bg-volume` | `0.10` | Background music volume (0.0–1.0) |
| `--no-bg-music` | — | Disable background music entirely |
| `--language` | `en` | Whisper transcription language |

## Pipeline

Narrated mode runs 9 steps; classic mode skips 4–6's narration work and runs the original 6.

1. **Cut & concat** — slices the timestamp ranges out of the source and concatenates them (ffmpeg).
2. **Layout** — classic: square center crop. Narrated: the square crop centred on a black 1080×1920 canvas (title card band above, narration band below).
3. **Dialogue captions** — from the movie's `.srt` when available (cues remapped onto the clip timeline, per-word timings synthesized); otherwise `faster-whisper` locally, seeded with the movie title to bias vocabulary toward proper nouns.
4. **Narration script** (narrated mode) — the clip's timed dialogue is grouped into *beats*, and Ollama writes one analytical present-tense line per beat (tension, subtext, character dynamics, drawing on its knowledge of the film). Each line is anchored to its beat's timestamp and saved to `narration.txt` as `[SS.s] line`.
5. **Voiceover** (narrated mode) — edge-tts reads each beat's line and places it at that beat's anchor time, so the narration spreads across the whole scene in sync with the dialogue instead of front-loading. A line is sped up (≤1.35×) only if it would overrun into the next beat.
6. **Metadata** — fetches a Wikipedia summary of the movie/show for context, then asks Ollama to write a title (≤40 chars, used as the on-screen title card), a description (opens with movie background, ends with hashtags), and 10–15 keyword tags. Hashtags are also guaranteed-appended programmatically so the LLM can't skip them.
7. **Captions** — builds an `.ass` subtitle file:
   - Words light up one at a time in sync with speech (karaoke-style), cycling through a TikTok-style color palette per sentence.
   - Narrated mode adds the narration text band beneath the video, one sentence at a time in sync with the voice.
   - Profanity is auto-censored (`shit` → `SH*T`, `fucking` → `F****G`, etc.) — edit the `_PROFANITY` set in `src/captions.py` to add/remove words.
   - A white-panel title card slides in at the top, holds ~1.5s, then slides off.
8. **Burn-in + audio mix** — ffmpeg bakes the captions in. Narrated mode mixes movie audio (gently side-chain ducked while the voice speaks) + narration at `--narration-volume` + optional bg music (off by default — three layers gets muddy).
9. **Thumbnail** (narrated mode) — a full-bleed 9:16 thumbnail (no black bars): a frame of the clip is cropped to fill the frame, with the title overlaid across the top, the movie name across the bottom, and a lined border. By default ffmpeg auto-picks the frame and centres the crop; the web UI lets you scrub to the exact frame and pan the 9:16 crop window (see below). Saved as `thumbnail.jpg`.

## Output

Everything lands in `output/<movie-slug>/`:

| File | Contents |
|------|----------|
| `clip_raw.mp4` | Cut/concatenated clip, widescreen, no captions |
| `clip_square.mp4` / `clip_vertical.mp4` | Formatted frame (square crop / square-on-black 9:16) |
| `captions.ass` | Generated subtitle file |
| `clip_final.mp4` | Final captioned vertical video — upload this |
| `metadata.json` | `{ title, description, tags }` |
| `description.txt` | Ready-to-paste YouTube description with hashtags |
| `narration.txt` / `narration.wav` | Narration script + voiceover track (narrated mode) |
| `thumbnail.jpg` | Representative frame with title overlay (narrated mode) |

## Web UI

A standalone form-based UI lives in `web/app.py` — it shells out to `src/main.py` the same way any other caller does, so the CLI itself is untouched.

```
pip install -r requirements.txt
python web/app.py
```

Opens on `http://0.0.0.0:8081` (override with `CLIPPER_WEB_HOST` / `CLIPPER_WEB_PORT`). Reach it from another device on your Tailscale network at `http://<this-machine>.<tailnet>.ts.net:8081`.

When you enter a movie path the UI checks for a subtitle file next to it. If found, you get:

- a **"Use this subtitle file for captions"** toggle (local processing only — untick to force Whisper), and
- a **"Suggest scenes from subtitles"** button with its own model picker. Suggestions render as clickable cards; each click fills a clip-range row, so you can stack several scenes into one Short. Suggestion defaults to a **local** model because the whole subtitle file goes into the prompt — the cloud option is labelled with its bandwidth cost and is never used automatically.

**Narrated runs are two-phase in the web UI** so you can choose the thumbnail *before* the slow render:

1. **① Cut clip & pick thumbnail** — cuts just the raw clip, then shows the **thumbnail picker**: the clip loads in a scrubber — pause on the frame you want, move the **crop slider** to choose which 9:16 slice of the (wider) frame to keep (a highlighted box shows the crop live), set the title/movie text, and click *Use this frame as thumbnail*.
2. **② Render final video** — runs the full pipeline, reusing that cut and your hand-picked thumbnail.

The chosen thumbnail is **baked in as the first 0.5 seconds of the final video**. YouTube's mobile app can't take a directly-uploaded thumbnail for a Short, but it lets you pick a frame — so scrub to the very start and that designed thumbnail frame is right there to select.

Pasting a movie path wrapped in quotes (e.g. Windows' *Copy as path*) works — the surrounding quotes are stripped automatically.

## Background music

Drop `.mp3` / `.wav` / `.ogg` files into the `audio/` folder. One is picked at random each run. Use `--bg-music` to pin a specific file, or `--no-bg-music` to skip.

## Notes

- The vertical crop is a simple center crop — no face tracking. For off-center subjects, pre-crop the source or adjust `--crop-w`/`--crop-h`.
- Wikipedia context lookup is best-effort — if the title isn't found, the LLM falls back to its own knowledge.
- For the best transcription accuracy on heavy accents or non-English dialogue, use `--whisper-model large-v3`.
