"""Explicit domain glossary for an opt-in cross-language retrieval experiment.

No generated answers, query IDs, gold pages, or corpus-specific boosts are used.
This is NOT a general translator. Users can inspect every matched term.
"""

from __future__ import annotations

import json
import unicodedata

from .parsing import digest

TERMS = (
    (("伊戈尔远征记", "伊戈尔远征纪", "伊戈里远征记"), "Слово о полку Игореве"),
    (("往年纪事", "往年记事"), "Повесть временных лет"),
    (("雅罗斯拉芙娜", "雅罗斯拉夫娜"), "Ярославна Ярославны"),
    (("莫诺马赫", "蒙诺马赫"), "Мономах Мономаха"),
    (("法律与神恩", "法律与恩典"), "Слово о законе и благодати"),
    (("图罗夫基里尔", "图罗夫的基里尔", "图罗夫斯基"), "Кирилл Туровский Туровского"),
    (("利哈乔夫", "利哈切夫"), "Лихачев Лихачева"),
    (("迪吉尼斯", "第吉尼斯"), "Дигенис Девгениево деяние"),
    (("古罗斯", "古俄罗斯"), "Древняя Русь древнерусская"),
    (("拜占庭",), "Византия византийское"),
    (("基辅",), "Киев Киевская"),
    (("索菲亚", "圣索菲亚"), "София Софии"),
    (("编年史", "编年"), "летопись летописание"),
    (("修道院",), "монастырь монастыри"),
    (("基督教化", "基督教"), "христианизация христианство"),
    (("教堂",), "храм собор"),
    (("世俗文学", "世俗"), "светская литература"),
    (("书面文化",), "книжная культура"),
    (("训诫", "教诲"), "Поучение"),
    (("诸王公", "王公"), "князья князей"),
    (("孩子",), "дети"),
    (("国家书信", "面向国家的书信"), "послание к нации"),
    (("哭诉", "哀歌", "哭泣"), "плач"),
    (("离别歌曲", "离别歌"), "песни разлуке"),
    (("纪念日",), "день памяти"),
    (("旧历",), "старый стиль"),
    (("体裁",), "жанр жанровый"),
    (("独特性", "文化独特"), "уникальность"),
    (("哲学",), "философия философской"),
    (("古代哲学", "古代思想"), "античная мысль"),
    (("文学",), "литература"),
    (("翻译", "传入"), "перевод переводная"),
    (("音乐",), "музыка музыкальная"),
    (("建筑",), "архитектура"),
)
GLOSSARY_SHA256 = digest(json.dumps(TERMS, ensure_ascii=False, separators=(",", ":")))


def rewrite_query(query: str) -> tuple[str, list[dict]]:
    text = unicodedata.normalize("NFC", query)
    matches = []
    for forms, target in TERMS:
        for form in forms:
            start = text.find(form)
            if start >= 0:
                matches.append((start, start + len(form), form, target))
    chosen, covered = [], set()
    # Longest phrase wins; avoid counting both a title and a shorter term inside it.
    for start, end, form, target in sorted(matches, key=lambda m: (-(m[1] - m[0]), m[0], m[2])):
        positions = set(range(start, end))
        if not positions & covered:
            chosen.append({"source": form, "target": target})
            covered.update(positions)
        if len(chosen) == 8:
            break
    rewritten = " ".join(dict.fromkeys(m["target"] for m in chosen)) if chosen else query
    return rewritten, chosen
