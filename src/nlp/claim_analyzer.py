import re


DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b(?:19|20)\d{2}\b",
    r"\b(?:today|yesterday|tomorrow|this year|last year|next year)\b",
]


CLAIM_TYPE_KEYWORDS = {
    "announcement": [
        "announced",
        "launch",
        "launched",
        "introduced",
        "unveiled",
        "revealed",
    ],
    "policy": [
        "policy",
        "law",
        "rule",
        "regulation",
        "scheme",
        "bill",
    ],
    "event": [
        "happened",
        "occurred",
        "won",
        "lost",
        "defeated",
        "elected",
    ],
    "death": [
        "died",
        "dead",
        "death",
        "killed",
    ],
    "appointment": [
        "appointed",
        "named",
        "selected",
        "promoted",
    ],
}


def extract_dates(text):
    dates = []

    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        dates.extend(matches)

    return list(dict.fromkeys(dates))


def detect_claim_type(text):
    text_lower = text.lower()

    for claim_type, keywords in CLAIM_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return claim_type

    return "general"


def extract_entities(text):
    """
    Lightweight entity extraction.

    Finds capitalized multi-word phrases and organizations.
    This is intentionally simple for the first version.
    """

    pattern = r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b"

    matches = re.findall(pattern, text)

    stop_words = {
    "The",
    "A",
    "An",
    "This",
    "That",
    "According",
}

    entities = []

    for item in matches:
        if item not in stop_words:
            entities.append(item)

    return list(dict.fromkeys(entities))


def analyze_claim(text):
    if not text or not text.strip():
        raise ValueError("Claim cannot be empty.")

    return {
        "claim": text.strip(),
        "entities": extract_entities(text),
        "claim_type": detect_claim_type(text),
        "dates": extract_dates(text),
    }


if __name__ == "__main__":

    claim = (
        "India announced a new artificial intelligence policy "
        "in 2026."
    )

    result = analyze_claim(claim)

    print("=" * 50)
    print("CLAIM ANALYSIS")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")
        