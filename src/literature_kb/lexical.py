from __future__ import annotations

import re
import unicodedata

TOKENIZER_VERSION = "nfc-casefold-han-bigram-word-v1"
PARTS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[^\W\d_]+|\d+", re.UNICODE)
HAN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")

# Explicit, finite domain aliases. This is query expansion, NOT multilingual embeddings,
# automatic translation, Russian lemmatization, or evidence that arbitrary Chinese queries work.
ALIASES = (
    (("伊戈尔远征记", "伊戈尔远征纪", "伊戈里远征记"), "слово полку игореве"),
    (("往年纪事", "往年记事"), "повесть временных лет"),
    (("莫诺马赫", "蒙诺马赫"), "мономах мономаха"),
    (("法律与神恩", "法律与恩典", "论法律与神恩"), "законе благодати иларион"),
    (("图罗夫", "图罗夫斯基"), "кирилл туровский туровского"),
    (("迪吉尼斯", "第吉尼斯"), "девгениево дигенис"),
    (("拜占庭",), "византия византийской византийская"),
    (("利哈乔夫", "利哈切夫"), "лихачев лихачева"),
)


def tokens(text: str) -> list[str]:
    parts = PARTS.findall(unicodedata.normalize("NFC", text).casefold().replace("ё", "е"))
    out = []
    for part in parts:
        if HAN.fullmatch(part) and len(part) > 1:
            out.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            out.append(part)
    return out


def query_expression(query: str, expand_aliases: bool) -> tuple[str, list[str]]:
    terms = tokens(query)
    expansions = []
    if expand_aliases:
        normalized = unicodedata.normalize("NFC", query).casefold()
        for forms, expanded in ALIASES:
            if any(form in normalized for form in forms):
                terms.extend(tokens(expanded))
                expansions.append(expanded)
    # No FTS operators from the user are interpolated. Bound growth and tokenize before quoting.
    unique = list(dict.fromkeys(terms))[:128]
    return " OR ".join('"' + term + '"' for term in unique), expansions
