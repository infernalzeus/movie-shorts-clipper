# movie-shorts-clipper

Turn a movie file into a captioned YouTube Short: pick one or more timestamp ranges,
cut/concat them, crop to vertical 9:16, transcribe with local Whisper, burn in
glowy gold word-pop captions plus a sliding title card, and generate a
title/description/tags with a local Ollama model.

## Pipeline

1. **Cut & concat** — `src/clip_selector.py` cuts the timestamp range(s) you give it
   out of the source movie and concatenates them into one clip (ffmpeg).
2. **Crop to vertical** — `src/reformat.py` center-crops the clip to 9:16 (default
   1080x1920) and scales it, since source movies are widescreen.
3. **Transcribe** — `src/transcriber.py` runs local `faster-whisper` on the cropped
   clip to get word-level timestamps.
4. **Metadata** — `src/metadata.py` sends the transcript, plus the movie/show name
   parsed from the source filename, to a local Ollama model and gets back a JSON
   title/description/tags for the Short (title is short — it doubles as the
   on-screen title card text).
5. **Captions** — `src/captions.py` builds an `.ass` subtitle file:
   - One bold gold word at a time pops in sync with speech, with a blurred
     glow/shine halo (two-layer ASS style: blurred glow + crisp text).
   - A white-box, black-text, all-caps title card slides in at the very top,
     holds for ~1.5s, then slides up off-screen by ~2s in.
6. **Burn-in** — `src/burn.py` burns the `.ass` captions into the video with
   ffmpeg's `libass`-backed `ass` filter.

## Requirements

- `ffmpeg` and `ffprobe` on PATH (with `libass` support — check via `ffmpeg -filters | grep ass`).
- A running local Ollama server (`http://localhost:11434`) with a model pulled.
  Default model is `nemotron-3-super:cloud`. Reasoning models are fine here (unlike
  some others, e.g. `qwen3.6`, which puts output in a `thinking` field instead of
  `response` under `format=json` and returns empty metadata — avoid those).
- Python deps: `pip install -r requirements.txt`

## Usage

```
cd src
python main.py
```

It will prompt for:
- the movie file path
- one or more clip ranges, e.g. `3:20-3:30; 5:15-5:45` (semicolon or comma separated)

Or skip the prompts with flags:

```
python main.py --video "C:\movies\The.Last.Heist.2019.1080p.BluRay.x264-GROUP.mkv" \
  --ranges "3:20-3:30; 5:15-5:45" \
  --whisper-model small --ollama-model nemotron-3-super:cloud \
  --crop-w 1080 --crop-h 1920
```

The movie/show name used for metadata is parsed automatically from the filename
(strips brackets, resolution/codec/release tags, years, release-group suffixes).
Override it with `--source-title "The Last Heist"` if the auto-parse looks off.

## Output

Everything lands in `output/<movie-name-slug>/`:

- `clip_raw.mp4` — cut/concatenated clip, no crop/captions
- `clip_vertical.mp4` — cropped to 9:16
- `captions.ass` — generated subtitle file (word-pop + title card)
- `clip_final.mp4` — final captioned vertical video
- `metadata.json` — `{title, description, tags}`
- `description.txt` — ready-to-paste YouTube description with hashtags

## Notes

- `--whisper-model` accepts any faster-whisper size (`tiny`, `base`, `small`,
  `medium`, `large-v3`, ...). Larger models are slower but more accurate.
- The vertical crop is a simple center crop — no subject tracking/face detection.
  For off-center subjects, pre-crop the source or adjust `--crop-w`/`--crop-h`.
- The Ollama title is capped at ~40 characters by the prompt so it fits the
  on-screen title card; it's reused as-is for the YouTube title.
