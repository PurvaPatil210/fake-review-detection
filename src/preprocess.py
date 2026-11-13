import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STOP_WORDS = ENGLISH_STOP_WORDS


def preprocess_text(text):
    """Normalize review text and remove common English stop words."""

    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [word for word in words if word not in _STOP_WORDS]

    return " ".join(words)
