# movie-shorts-clipper

Turn any movie or TV file into a upload-ready YouTube Short. Give it a file and one or more timestamp ranges — it cuts the clip, crops to 9:16, transcribes with local Whisper, burns in animated word-pop captions, and generates a YouTube title/description/tags via a local Ollama model.

## What you need installed first

- **ffmpeg** on PATH, compiled with `libass` — verify with `ffmpeg -filters | grep ass`
- **Ollama** running at `http://localhost:11434` with a model pulled. Default: `nemotron-3-super:cloud`.
  - Avoid reasoning models like `qwen3.6` — they return output in a `thinking` field under `format=json`, which breaks metadata parsing.
- **Python 3.11+** and dependencies: `pip install -r requirements.txt`
- **faster-whisper** will download the Whisper model on first run (~1.5 GB for `medium`). No manual setup needed.

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

## All flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--video` | (prompt) | Path to source movie file |
| `--ranges` | (prompt) | Clip ranges, e.g. `3:20-3:30; 5:15-5:45` |
| `--source-title` | parsed from filename | Movie/show name for metadata — override if auto-parse looks off |
| `--whisper-model` | `medium` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--ollama-model` | `nemotron-3-super:cloud` | Ollama model for title/description/tags |
| `--crop-w` / `--crop-h` | `1080` / `1920` | Output resolution (default 9:16 vertical) |
| `--bg-music` | random from `audio/` | Path to a background music file |
| `--bg-volume` | `0.10` | Background music volume (0.0–1.0) |
| `--no-bg-music` | — | Disable background music entirely |
| `--language` | `en` | Whisper transcription language |

## Pipeline

1. **Cut & concat** — slices the timestamp ranges out of the source and concatenates them (ffmpeg).
2. **Crop to vertical** — center-crops to 9:16 with a blurred background fill for the pillarbox areas.
3. **Transcribe** — runs `faster-whisper` locally with `beam_size=5` and accent-retry temperatures. Seeded with the movie title to bias vocabulary toward proper nouns. Default model: `medium`.
4. **Metadata** — fetches a Wikipedia summary of the movie/show for context, then asks Ollama to write a title (≤40 chars, used as the on-screen title card), a description (opens with movie background, ends with hashtags), and 10–15 keyword tags. Hashtags are also guaranteed-appended programmatically so the LLM can't skip them.
5. **Captions** — builds an `.ass` subtitle file:
   - Words light up one at a time in sync with speech (karaoke-style), cycling through a TikTok-style color palette per sentence.
   - Profanity is auto-censored (`shit` → `SH*T`, `fucking` → `F****G`, etc.) — edit the `_PROFANITY` set in `src/captions.py` to add/remove words.
   - A white-panel title card slides in at the top, holds ~1.5s, then slides off.
6. **Burn-in** — ffmpeg bakes the captions into the final video. Optional background music is mixed in at low volume.

## Output

Everything lands in `output/<movie-slug>/`:

| File | Contents |
|------|----------|
| `clip_raw.mp4` | Cut/concatenated clip, widescreen, no captions |
| `clip_square.mp4` | Cropped to 9:16 with blurred background |
| `captions.ass` | Generated subtitle file |
| `clip_final.mp4` | Final captioned vertical video — upload this |
| `metadata.json` | `{ title, description, tags }` |
| `description.txt` | Ready-to-paste YouTube description with hashtags |

## Background music

Drop `.mp3` / `.wav` / `.ogg` files into the `audio/` folder. One is picked at random each run. Use `--bg-music` to pin a specific file, or `--no-bg-music` to skip.

## Notes

- The vertical crop is a simple center crop — no face tracking. For off-center subjects, pre-crop the source or adjust `--crop-w`/`--crop-h`.
- Wikipedia context lookup is best-effort — if the title isn't found, the LLM falls back to its own knowledge.
- For the best transcription accuracy on heavy accents or non-English dialogue, use `--whisper-model large-v3`.
