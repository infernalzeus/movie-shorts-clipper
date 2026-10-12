"""Recover usable data from JSON an LLM didn't finish writing.

Every agent in this pipeline asks a model for JSON, and models — small local ones
especially — routinely stop mid-answer: they write a dozen good array entries and
then degenerate into repetition or whitespace, or they cut off in the middle of a
long string. `json.loads` rejects the whole response, and everything the model DID
get right is thrown away with it.

Two shapes to recover, one function each:

  loads_lenient(raw, key)  {"beats": [ {...}, {...}, <cut> ]}  -> the complete entries
  salvage_object(raw)      {"title": "...", "description": <cut>} -> the complete fields

Neither ever raises: they return [] / {} when there is genuinely nothing to keep,
so callers can decide whether that is fatal. It usually isn't.
"""

import json
import re

__all__ = ["loads_lenient", "salvage_object", "trim_to_sentence"]


def _scan(raw: str, start: int):
    """Yield (index, char, depth, in_string) walking JSON text from `start`.

    Depth counts open {} and [] together; strings (and their escapes) are tracked
    so braces inside a string value never move the depth.
    """
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            yield i, ch, depth, True
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        yield i, ch, depth, False


def loads_lenient(raw: str, key: str) -> list[dict]:
    """Parse {"<key>": [...]} , keeping every COMPLETE object from a cut-off array."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get(key), list):
            return [o for o in parsed[key] if isinstance(o, dict)]
    except json.JSONDecodeError:
        pass

    k = raw.find(f'"{key}"')
    if k < 0:
        return []
    arr = raw.find("[", k)
    if arr < 0:
        return []

    out: list[dict] = []
    obj_start = None
    for i, ch, depth, in_str in _scan(raw, arr):
        if in_str:
            continue
        if ch == "{" and depth == 2:      # array is depth 1, its objects depth 2
            obj_start = i
        elif ch == "}" and depth == 1 and obj_start is not None:
            try:
                obj = json.loads(raw[obj_start:i + 1])
                if isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError:
                pass
            obj_start = None
    return out


def salvage_object(raw: str) -> dict:
    """Parse a single JSON object, keeping the fields that finished.

    Truncation lands mid-value, so the last complete field is the one before the
    last top-level comma. Cut there, close the brace, and parse — retrying at
    earlier commas if the first attempt still fails.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    if start < 0:
        return {}

    # First try to CLOSE what is open rather than discard it: a response cut
    # mid-description still holds most of that description, and closing the
    # string keeps it (trim_to_sentence tidies the ragged end).
    stack: list[str] = []
    ended_in_string = False
    for _i, ch, _depth, in_str in _scan(raw, start):
        ended_in_string = in_str
        if in_str:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()
    if stack:
        body = raw[start:].rstrip()
        if body.endswith("\\"):        # a dangling escape would swallow our quote
            body = body[:-1]
        if ended_in_string:
            body += '"'
        body = body.rstrip().rstrip(",")
        try:
            obj = json.loads(body + "".join(reversed(stack)))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Otherwise fall back to the last complete field: cut at a top-level comma.
    cuts = [i for i, ch, depth, in_str in _scan(raw, start)
            if not in_str and ch == "," and depth == 1]
    for cut in reversed(cuts):
        try:
            obj = json.loads(raw[start:cut] + "}")
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*(?=\s|$)")


def trim_to_sentence(text: str) -> str:
    """Cut a salvaged string back to its last complete sentence.

    A description recovered from a truncated response usually ends mid-word; a
    clean stop reads as deliberate rather than broken.
    """
    text = (text or "").strip()
    if not text:
        return ""
    ends = list(_SENTENCE_END.finditer(text))
    if not ends:
        return text
    return text[:ends[-1].end()].strip()
