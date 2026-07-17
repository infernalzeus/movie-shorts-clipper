"""Generate an .ass subtitle file:
- Title card: white full-width panel at the top that slides off after ~2 s.
- Captions: sentence-by-sentence, each word lights up (karaoke) as it is spoken,
  sentence fades in and fades out smoothly.
- Narration (optional): TTS voiceover text shown in the band beneath the video
  on the vertical layout, one sentence at a time in sync with the voice.
"""

import re
from pathlib import Path

from transcriber import Word

# ── Profanity censorship ─────────────────────────────────────────────────────
# Maps the lowercase word (no punctuation) → censored display form.
# Keeps first and last letter, asterisks in the middle, preserves surrounding punctuation.

_PROFANITY = {
    "shit", "shits", "shitting", "shitted",
    "fuck", "fucks", "fucking", "fucked", "fucker", "fuckers", "motherfucker", "motherfuckers",
    "bitch", "bitches", "bitching",
    "ass", "asses", "asshole", "assholes",
    "bastard", "bastards",
    "cunt", "cunts",
    "damn", "damned", "goddamn", "goddamned",
    "hell",  # optional — comment out if too aggressive
    "piss", "pissed", "pissing",
    "cock", "cocks", "dick", "dicks",
    "whore", "whores",
    "crap", "craps",
}


def _censor_word(token: str) -> str:
    """Return censored form of token if it contains profanity, preserving surrounding punctuation."""
    # split leading/trailing punctuation from the core word
    m = re.fullmatch(r"([^a-zA-Z]*)([a-zA-Z]+)([^a-zA-Z]*)", token)
    if not m:
        return token
    pre, word, post = m.group(1), m.group(2), m.group(3)
    if word.lower() not in _PROFANITY:
        return token
    if len(word) <= 2:
        return pre + word[0] + "*" + post
    stars = "*" * (len(word) - 2)
    censored = word[0] + stars + word[-1]
    # preserve original casing of first/last letter
    return pre + censored + post

# ── Layer ordering (higher = drawn on top) ──────────────────────────────────
_LAYER_TITLE_BG   = 0   # white rect drawing
_LAYER_TITLE_TEXT = 1   # black title text
_LAYER_CAPTION    = 2   # sentence captions
_LAYER_NARRATION  = 3   # narration text band

# ── Colours (ASS ABGR hex: &HAABBGGRR) ──────────────────────────────────────
_WHITE       = "&H00FFFFFF"
_BLACK       = "&H00000000"
_INVISIBLE   = "&HFF000000"  # fully transparent → unspoken words stay hidden

# TikTok-style caption palette (ASS is &H00BBGGRR — no alpha, opaque). Cycled per sentence.
_CAPTION_PALETTE = [
    "&H00FFFFFF",  # white
    "&H0000FFFF",  # yellow
    "&H0000FF00",  # green (lime)
    "&H00FFFF00",  # cyan
    "&H00FF66FF",  # pink/magenta
    "&H0000A5FF",  # orange
]

_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TitleBg,Arial,20,{white},{white},{white},{white},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: TitleText,Arial Black,{title_fontsize},{black},{black},{white},{black},-1,0,0,0,100,100,0,0,1,0,0,5,40,40,40,1
Style: Caption,Arial Black,{caption_fontsize},{white},{invisible},{black},{black},-1,0,0,0,100,100,0,0,1,4,0,2,60,60,{caption_margin_v},1
Style: Narration,Arial,{narration_fontsize},{narration_color},{narration_color},{black},{black},0,0,0,0,100,100,0,0,1,1,0,2,70,70,{narration_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    cs = round(seconds * 100)
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs  = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ── Title card ───────────────────────────────────────────────────────────────

def _title_card_lines(
    title: str,
    video_width: int,
    panel_h: int,
    hold_ms: int = 1500,
    slide_ms: int = 500,
) -> list[str]:
    """White full-width panel + centred black title, both sliding off the top together."""
    end_time = (hold_ms + slide_ms) / 1000
    t1, t2 = hold_ms, hold_ms + slide_ms
    cx = video_width // 2
    y_text = panel_h // 2  # vertical centre of panel

    # Drawing: full-width white rectangle at top (an7 = top-left anchor at pos 0,0)
    rect = f"m 0 0 l {video_width} 0 {video_width} {panel_h} 0 {panel_h}"
    bg_override = (
        f"{{\\an7\\move(0,0,0,{-panel_h},{t1},{t2})"
        f"\\p1\\c&HFFFFFF&}}{rect}{{\\p0}}"
    )

    text = title.upper().replace("\\", "").replace("{", "").replace("}", "")
    text_override = (
        f"{{\\an5\\move({cx},{y_text},{cx},{-y_text},{t1},{t2})"
        f"\\c&H000000&\\bord0\\shad0}}{text}"
    )

    return [
        f"Dialogue: {_LAYER_TITLE_BG},{_fmt_time(0)},{_fmt_time(end_time)},TitleBg,,0,0,0,,{bg_override}",
        f"Dialogue: {_LAYER_TITLE_TEXT},{_fmt_time(0)},{_fmt_time(end_time)},TitleText,,0,0,0,,{text_override}",
    ]


# ── Sentence grouping ────────────────────────────────────────────────────────

_MAX_WORDS_PER_LINE = 7  # force a break even mid-sentence so captions don't sprawl across the screen


def _group_sentences(
    words: list[Word], pause_threshold: float = 0.65, max_words: int = _MAX_WORDS_PER_LINE,
) -> list[list[Word]]:
    """Split a flat word list into short caption chunks by punctuation, long pauses, or a word-count cap."""
    sentences: list[list[Word]] = []
    current: list[Word] = []

    for i, word in enumerate(words):
        if not word.text:
            continue
        current.append(word)
        stripped = word.text.strip()
        ends_sentence = bool(stripped) and stripped[-1] in ".?!"
        is_last       = (i == len(words) - 1)
        long_pause    = (not is_last) and (words[i + 1].start - word.end > pause_threshold)
        too_long      = len(current) >= max_words

        if ends_sentence or long_pause or too_long or is_last:
            if current:
                sentences.append(current)
            current = []

    return sentences


# ── Caption event builder ────────────────────────────────────────────────────

def _sentence_line(sentence_words: list[Word], color: str, fade_out_ms: int = 400) -> str:
    """One ASS Dialogue line for an entire sentence with per-word karaoke timing.

    color overrides PrimaryColour for this line (TikTok-style per-sentence color).
    A glow effect comes from \\blur softening the (already-set) style outline.
    """
    evt_start = sentence_words[0].start
    evt_end   = sentence_words[-1].end + fade_out_ms / 1000

    parts: list[str] = []
    for k, word in enumerate(sentence_words):
        # karaoke duration: from this word's start to the next word's start (or word end)
        if k + 1 < len(sentence_words):
            dur_s = sentence_words[k + 1].start - word.start
        else:
            dur_s = word.end - word.start
        dur_cs = max(1, round(dur_s * 100))

        text = _censor_word(word.text.strip()).upper().replace("\\", "").replace("{", "").replace("}", "")
        parts.append(f"{{\\k{dur_cs}}}{text}")

    # \fad = fade in/out, \c = per-line text color, \blur = soft glow on the outline
    body = "{\\fad(250," + str(fade_out_ms) + f")\\c{color}\\blur2" + "}" + " ".join(parts)
    return (
        f"Dialogue: {_LAYER_CAPTION},"
        f"{_fmt_time(evt_start)},{_fmt_time(evt_end)},"
        f"Caption,,0,0,0,,{body}"
    )


# ── Narration band ───────────────────────────────────────────────────────────

_NARRATION_COLOR = "&H00F2F0EE"  # soft off-white


def _narration_line(segment) -> str:
    """One ASS Dialogue line for a narration sentence (plain text, gentle fade).

    segment is any object with .text/.start/.end (tts.NarrationSegment).
    """
    text = segment.text.replace("\\", "").replace("{", "").replace("}", "")
    body = "{\\fad(200,200)\\blur0.6}" + text
    return (
        f"Dialogue: {_LAYER_NARRATION},"
        f"{_fmt_time(segment.start)},{_fmt_time(segment.end + 0.25)},"
        f"Narration,,0,0,0,,{body}"
    )


# ── Public API ───────────────────────────────────────────────────────────────

def build_ass(
    words: list[Word],
    output_path: Path,
    video_width: int = 1080,
    video_height: int = 1920,
    title: str | None = None,
    caption_fontsize: int | None = None,
    title_fontsize: int | None = None,
    narration: list | None = None,
    layout: dict | None = None,
) -> Path:
    """Write an .ass file with a white-panel title card, sentence-level karaoke
    captions, and (optionally) the narration text band.

    layout comes from reformat.compose_vertical and pins the caption/narration
    geometry to the actual video block; without it the classic centred-square
    geometry is assumed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    caption_fontsize  = caption_fontsize  or max(52, video_width // 18)
    title_fontsize    = title_fontsize    or max(44, video_width // 22)

    if layout:
        video_top    = layout["video_top"]
        video_bottom = layout["video_bottom"]
        # Panel fits in the blurred strip above the video block
        panel_h = max(100, min(video_top - 20, video_width // 9))
        # Dialogue captions sit ~100px above the bottom edge of the video — same
        # placement as the classic square pipeline
        caption_margin_v = video_height - video_bottom + 100
        # Narration is anchored to the BOTTOM of the screen (an2) — multi-line
        # text grows upward into the black band rather than crowding the video.
        narration_margin_v = 70
    else:
        # The square movie occupies the vertical centre of the frame.
        sq_offset = (video_height - video_width) // 2   # y where square starts (e.g. 420 for 1080x1920)
        panel_h = max(100, min(sq_offset - 20, video_width // 9))
        # Caption bottom sits ~100px above the bottom edge of the square
        caption_margin_v = sq_offset + 100
        narration_margin_v = 70

    lines = [
        _HEADER.format(
            width=video_width,
            height=video_height,
            white=_WHITE,
            black=_BLACK,
            invisible=_INVISIBLE,
            title_fontsize=title_fontsize,
            caption_fontsize=caption_fontsize,
            caption_margin_v=caption_margin_v,
            narration_fontsize=max(38, video_width // 26),
            narration_color=_NARRATION_COLOR,
            narration_margin_v=narration_margin_v,
        )
    ]

    if title:
        lines.extend(_title_card_lines(title, video_width, panel_h))

    for i, sentence_words in enumerate(_group_sentences(words)):
        color = _CAPTION_PALETTE[i % len(_CAPTION_PALETTE)]
        lines.append(_sentence_line(sentence_words, color))

    for segment in narration or []:
        lines.append(_narration_line(segment))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
