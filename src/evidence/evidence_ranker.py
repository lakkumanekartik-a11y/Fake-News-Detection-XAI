import re
from datetime import datetime
from urllib.parse import urlparse


TRUSTED_DOMAINS = {
    "gov.in": 1.0,
    "pib.gov.in": 1.0,
    "who.int": 1.0,
    "un.org": 1.0,
    "reuters.com": 0.95,
    "bbc.com": 0.90,
    "apnews.com": 0.90,
    "theguardian.com": 0.85,
}


def tokenize(text):
    return set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            text.lower(),
        )
    )


def relevance_score(claim, evidence):
    claim_words = tokenize(claim)

    evidence_text = " ".join([
        evidence.get("title", ""),
        evidence.get("content", ""),
    ])

    evidence_words = tokenize(evidence_text)

    if not claim_words:
        return 0.0

    return len(claim_words & evidence_words) / len(claim_words)


def entity_match_score(claim, evidence):
    claim_words = tokenize(claim)

    evidence_text = " ".join([
        evidence.get("title", ""),
        evidence.get("content", ""),
    ]).lower()

    important_words = [
        word for word in claim_words
        if len(word) >= 5
    ]

    if not important_words:
        return 0.0

    matches = sum(
        1
        for word in important_words
        if word in evidence_text
    )

    return matches / len(important_words)


def source_quality(url):
    if not url:
        return 0.3

    domain = urlparse(url).netloc.lower()
    domain = domain.replace("www.", "")

    for trusted, score in TRUSTED_DOMAINS.items():
        if domain == trusted or domain.endswith("." + trusted):
            return score

    return 0.5


def extract_year(text):
    matches = re.findall(
        r"\b(?:19|20)\d{2}\b",
        text,
    )

    if matches:
        return int(matches[-1])

    return None


def date_relevance(claim, evidence):
    claim_year = extract_year(claim)

    evidence_text = " ".join([
        evidence.get("title", ""),
        evidence.get("content", ""),
        evidence.get("published_date", ""),
        evidence.get("date", ""),
    ])

    evidence_year = extract_year(evidence_text)

    if claim_year is None or evidence_year is None:
        return 0.5

    difference = abs(claim_year - evidence_year)

    if difference == 0:
        return 1.0

    if difference == 1:
        return 0.7

    if difference == 2:
        return 0.4

    return 0.2


def rank_evidence(claim, results):
    ranked = []

    for result in results:

        relevance = relevance_score(
            claim,
            result,
        )

        entity_match = entity_match_score(
            claim,
            result,
        )

        quality = source_quality(
            result.get("url", ""),
        )

        date_score = date_relevance(
            claim,
            result,
        )

        score = (
            0.40 * relevance
            + 0.30 * entity_match
            + 0.20 * quality
            + 0.10 * date_score
        )

        item = result.copy()

        item["relevance"] = relevance
        item["entity_match"] = entity_match
        item["source_quality"] = quality
        item["date_relevance"] = date_score
        item["evidence_score"] = score

        ranked.append(item)

    ranked.sort(
        key=lambda x: x["evidence_score"],
        reverse=True,
    )

    return ranked