"""Extract stable answer text from an incomplete JSON object during generation."""

from __future__ import annotations

import json


def answer_prefix(raw: str) -> str:
    decoder = json.JSONDecoder()
    position = len(raw) - len(raw.lstrip())
    if raw[position : position + 1] != "{":
        return ""
    position += 1
    try:
        while position < len(raw):
            position += len(raw[position:]) - len(raw[position:].lstrip())
            key, position = decoder.raw_decode(raw, position)
            if not isinstance(key, str):
                return ""
            position += len(raw[position:]) - len(raw[position:].lstrip())
            if raw[position : position + 1] != ":":
                return ""
            position += 1
            position += len(raw[position:]) - len(raw[position:].lstrip())
            if key == "answer":
                return _string_prefix(raw, position)
            _, position = decoder.raw_decode(raw, position)
            position += len(raw[position:]) - len(raw[position:].lstrip())
            if raw[position : position + 1] != ",":
                return ""
            position += 1
    except (ValueError, IndexError):
        return ""
    return ""


def _string_prefix(raw: str, start: int) -> str:
    if raw[start : start + 1] != '"':
        return ""
    position = start + 1
    safe_end = position
    while position < len(raw):
        character = raw[position]
        if character == '"':
            break
        if character == "\\":
            if position + 1 >= len(raw):
                break
            if raw[position + 1] == "u":
                if position + 6 > len(raw):
                    break
                try:
                    code = int(raw[position + 2 : position + 6], 16)
                    width = 6
                    if 0xD800 <= code <= 0xDBFF:
                        if raw[position + 6 : position + 8] != "\\u" or position + 12 > len(raw):
                            break
                        low = int(raw[position + 8 : position + 12], 16)
                        if not 0xDC00 <= low <= 0xDFFF:
                            break
                        width = 12
                    elif 0xDC00 <= code <= 0xDFFF:
                        break
                except ValueError:
                    break
                position += width
            elif raw[position + 1] in '"\\/bfnrt':
                position += 2
            else:
                break
        elif ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF:
            break
        else:
            position += 1
        safe_end = position
    return json.loads(raw[start:safe_end] + '"')
