import re
import unicodedata


def prepare_text(text: str) -> str:
    """Normalize encoder input only; never rewrite stored source/offsets."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).replace("\u00ad", "")).strip()
