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

# Caption font rotation. Cycled per sentence when caption_font_shuffle is on —
# the same idea as the colour palette, applied to the typeface. Every family
# here ships with Windows, so an .ass referencing it always resolves.
_CAPTION_FONTS = [
    "Anton",          # bundled (assets/fonts) — the viral-shorts caption face
    "Bebas Neue",     # bundled — tall condensed caps, the other ubiquitous one
    "Impact",
    "Arial Black",
    "Verdana",
    "Trebuchet MS",
    "Tahoma",
    "Georgia",
    "Comic Sans MS",
]

# Named colours (BBGGRR) for the non-shuffle case — one fixed caption colour.
_CAPTION_COLORS = {
    "white":  "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "green":  "&H0000FF00",
    "cyan":   "&H00FFFF00",
    "pink":   "&H00FF66FF",
    "orange": "&H0000A5FF",
}
CAPTION_ANIMATIONS = ("karaoke", "fade", "pop")

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
Style: Caption,{caption_font},{caption_fontsize},{white},{invisible},{black},{black},-1,{caption_italic},0,0,100,100,0,0,1,4,0,2,60,60,{caption_margin_v},1
Style: Narration,Arial,{narration_fontsize},{narration_color},{narration_color},{black},{black},0,{caption_italic},0,0,100,100,0,0,1,{narration_outline},{narration_shadow},2,{narration_margin_h},{narration_margin_h},{narration_margin_v},1

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

    text = title.replace("\\", "").replace("{", "").replace("}", "")   # respect typed case (no auto-uppercase)
    text_override = (
        f"{{\\an5\\move({cx},{y_text},{cx},{-y_text},{t1},{t2})"
        f"\\c&H000000&\\bord0\\shad0}}{text}"
    )

    return [
        f"Dialogue: {_LAYER_TITLE_BG},{_fmt_time(0)},{_fmt_time(end_time)},TitleBg,,0,0,0,,{bg_override}",
        f"Dialogue: {_LAYER_TITLE_TEXT},{_fmt_time(0)},{_fmt_time(end_time)},TitleText,,0,0,0,,{text_override}",
    ]


# ── Sentence grouping ────────────────────────────────────────────────────────

_MAX_WORDS_PER_LINE = 7  # cap so captions don't sprawl across the screen
_MIN_WORDS_PER_LINE = 3  # never strand a 1-2 word sliver on its own line

# Words a caption line should break BEFORE, not after — a new clause reads better
# starting a fresh line. Used by the semantic line-breaker below.
_CLAUSE_STARTERS = {
    "and", "but", "or", "so", "because", "when", "while", "which", "that",
    "who", "if", "then", "yet", "for", "nor", "as", "since", "though", "although",
    "before", "after", "until", "unless", "whereas",
}
# A word ending in one of these closes a clause — a good place to break AFTER.
_CLAUSE_ENDERS = ",;:—"


def _group_sentences(
    words: list[Word], pause_threshold: float = 0.65,
    max_words: int = _MAX_WORDS_PER_LINE, min_words: int = _MIN_WORDS_PER_LINE,
) -> list[list[Word]]:
    """Split a flat word list into short caption chunks.

    Hard boundaries: sentence-ending punctuation, a long spoken pause, or the end
    of the list. When a chunk merely hits the word cap, break SEMANTICALLY instead
    of chopping mid-phrase — at the latest clause boundary (a comma/colon/dash, or
    just before a conjunction). Borrowed from hotclip's (xixihhhh) caption
    line-breaking, which "respects real clause structures". Falls back to the hard
    cap only when no clause boundary is available.
    """
    sentences: list[list[Word]] = []
    current: list[Word] = []

    def flush(upto: int | None = None) -> None:
        nonlocal current
        if upto is None or upto >= len(current):
            if current:
                sentences.append(current)
            current = []
        elif upto > 0:
            sentences.append(current[:upto])
            current = current[upto:]

    for i, word in enumerate(words):
        if not word.text:
            continue
        current.append(word)
        stripped = word.text.strip()
        ends_sentence = bool(stripped) and stripped[-1] in ".?!"
        is_last       = (i == len(words) - 1)
        long_pause    = (not is_last) and (words[i + 1].start - word.end > pause_threshold)

        if ends_sentence or long_pause or is_last:
            flush()
            continue

        if len(current) >= max_words:
            # Find the latest clause boundary that leaves at least min_words behind.
            brk: int | None = None
            for j in range(len(current) - 1, min_words - 2, -1):
                w = current[j].text.strip()
                nxt = current[j + 1].text.strip().lower().strip(".,;:!?—-") if j + 1 < len(current) else ""
                if w and w[-1] in _CLAUSE_ENDERS:
                    brk = j + 1
                    break
                if nxt in _CLAUSE_STARTERS:
                    brk = j + 1
                    break
            flush(brk)   # brk is None → no clean boundary, take the hard cap

    return sentences


# ── Caption event builder ────────────────────────────────────────────────────

def _clean(word: Word) -> str:
    return _censor_word(word.text.strip()).upper().replace("\\", "").replace("{", "").replace("}", "")


def _sentence_line(
    sentence_words: list[Word], color: str, fade_out_ms: int = 400, animation: str = "karaoke",
    font: str | None = None,
) -> str:
    """One ASS Dialogue line for a whole sentence.

    color overrides PrimaryColour for this line. `animation` picks the effect:
    - karaoke: words light up one-by-one as spoken (\\k per word)
    - fade:    the whole sentence fades in/out together
    - pop:     the whole sentence scales up as it fades in
    font, when given, overrides the Caption style's family for this line only —
    that is how the per-sentence font rotation works.
    A glow comes from \\blur softening the style outline.
    """
    evt_start = sentence_words[0].start
    evt_end   = sentence_words[-1].end + fade_out_ms / 1000
    # An inline font override runs until the next backslash or brace, so a
    # family with spaces in its name needs no quoting.
    font_tag = ("\\" + "fn" + font) if font else ""

    if animation == "karaoke":
        parts: list[str] = []
        for k, word in enumerate(sentence_words):
            # karaoke duration: this word's start to the next word's start (or word end)
            if k + 1 < len(sentence_words):
                dur_s = sentence_words[k + 1].start - word.start
            else:
                dur_s = word.end - word.start
            dur_cs = max(1, round(dur_s * 100))
            parts.append(f"{{\\k{dur_cs}}}{_clean(word)}")
        body = "{\\fad(250," + str(fade_out_ms) + f")\\c{color}{font_tag}\\blur2" + "}" + " ".join(parts)
    else:
        text = " ".join(_clean(w) for w in sentence_words)
        if animation == "pop":
            override = ("{\\fad(150," + str(fade_out_ms)
                       + f")\\c{color}{font_tag}\\blur2\\fscx60\\fscy60\\t(0,160,\\fscx100\\fscy100)" + "}")
        else:  # "fade" (or any unknown) → plain fade, all words visible at once
            override = "{\\fad(250," + str(fade_out_ms) + f")\\c{color}{font_tag}\\blur2" + "}"
        body = override + text

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
    caption_font: str = "Arial Black",
    caption_animation: str = "karaoke",
    caption_shuffle: bool = True,
    caption_color: str = "white",
    caption_font_shuffle: bool = False,
    caption_italic: bool = False,
    shape: str = "vertical",
) -> Path:
    """Write an .ass file with a white-panel title card, sentence-level karaoke
    captions, and (optionally) the narration text.

    layout comes from reformat.compose_vertical and pins the caption/narration
    geometry to the actual video block; without it the geometry is derived from
    `shape` ("vertical" / "square" / "landscape").

    On landscape there is no black band to put text in, so the narration renders
    as a LOWER THIRD over the footage — bigger, with a heavier outline and a drop
    shadow so it stays readable against moving picture.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_landscape = shape == "landscape" or (layout is None and video_width > video_height)

    # Type size scales with the frame's SHORT side. Keying it to the width alone
    # gave a 1920-wide landscape frame a 106px caption — sized for a 1080-wide
    # vertical one.
    short_side = min(video_width, video_height)
    caption_fontsize  = caption_fontsize  or max(44, short_side // 18)
    title_fontsize    = title_fontsize    or max(40, short_side // 22)

    if is_landscape:
        # Full-bleed 16:9: no band above or below, so text sits over the picture.
        panel_h = max(90, min(video_height // 8, 170))
        caption_margin_v = 80
        narration_margin_v = 70
        narration_fontsize = max(40, short_side // 22)
        narration_margin_h = 180        # keep the lower third off the far edges
        narration_outline, narration_shadow = 3, 1
    elif layout:
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
        narration_fontsize = max(38, video_width // 26)
        narration_margin_h = 70
        narration_outline, narration_shadow = 1, 0
    else:
        # The square movie occupies the vertical centre of the frame.
        sq_offset = max(0, (video_height - video_width) // 2)
        panel_h = max(100, min(sq_offset - 20, video_width // 9)) if sq_offset > 120 else max(90, video_width // 9)
        # Caption bottom sits ~100px above the bottom edge of the square
        caption_margin_v = sq_offset + 100 if sq_offset else 90
        narration_margin_v = 70
        narration_fontsize = max(38, video_width // 26)
        narration_margin_h = 70
        narration_outline, narration_shadow = 1, 0

    lines = [
        _HEADER.format(
            width=video_width,
            height=video_height,
            white=_WHITE,
            black=_BLACK,
            invisible=_INVISIBLE,
            title_fontsize=title_fontsize,
            caption_font=caption_font,
            caption_fontsize=caption_fontsize,
            caption_italic=(-1 if caption_italic else 0),
            caption_margin_v=caption_margin_v,
            narration_fontsize=narration_fontsize,
            narration_color=_NARRATION_COLOR,
            narration_margin_v=narration_margin_v,
            narration_margin_h=narration_margin_h,
            narration_outline=narration_outline,
            narration_shadow=narration_shadow,
        )
    ]

    if title:
        lines.extend(_title_card_lines(title, video_width, panel_h))

    fixed_color = _CAPTION_COLORS.get(caption_color, _WHITE)
    # Font rotation starts on the family the user picked, then cycles the rest,
    # so sentence 1 always looks like what the form promised.
    font_cycle = [caption_font] + [f for f in _CAPTION_FONTS if f != caption_font]
    for i, sentence_words in enumerate(_group_sentences(words)):
        color = _CAPTION_PALETTE[i % len(_CAPTION_PALETTE)] if caption_shuffle else fixed_color
        font = font_cycle[i % len(font_cycle)] if caption_font_shuffle else None
        lines.append(_sentence_line(sentence_words, color, animation=caption_animation, font=font))

    for segment in narration or []:
        lines.append(_narration_line(segment))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
