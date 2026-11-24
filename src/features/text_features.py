"""Simple numeric text features, shared by drift monitoring (and later the feature store).

They are computed from the cleaned text, so training-time and serving-time values match.
"""

FEATURE_NAMES = (
    "char_length",
    "word_count",
    "avg_word_length",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
)


def extract_features(text: str) -> dict[str, float]:
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    return {
        "char_length": float(len(text)),
        "word_count": float(len(words)),
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 3) if words else 0.0,
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "uppercase_ratio": round(sum(c.isupper() for c in letters) / len(letters), 4) if letters else 0.0,
    }
