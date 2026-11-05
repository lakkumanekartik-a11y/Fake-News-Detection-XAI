import re


STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "have",
    "has",
    "been",
    "was",
    "were",
    "are",
    "new",
    "announced",
    "announce",
    "announces",
    "launch",
    "launched",
    "introduce",
    "introduced",
}


def tokenize(text):

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
    }


def keyword_overlap(
    claim,
    evidence,
):

    claim_words = tokenize(
        claim
    )

    evidence_text = " ".join(
        [
            evidence.get("title", ""),
            evidence.get("content", ""),
        ]
    )

    evidence_words = tokenize(
        evidence_text
    )

    if not claim_words:
        return 0.0

    return (
        len(
            claim_words.intersection(
                evidence_words
            )
        )
        / len(claim_words)
    )


def filter_evidence(
    claim,
    results,
    min_overlap=0.30,
):

    filtered = []

    for result in results:

        score = keyword_overlap(
            claim,
            result,
        )

        item = result.copy()

        item["keyword_overlap"] = score

        if score >= min_overlap:

            filtered.append(
                item
            )

    return filtered