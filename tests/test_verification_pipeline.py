from src.evidence.evidence_filter import filter_evidence
from src.evidence.evidence_ranker import rank_evidence
from src.evidence.nli_verifier import verify_claim
from src.evidence.evidence_aggregator import aggregate_evidence
from src.decision.decision_engine import decide_verdict


def run_pipeline(claim, evidence):

    filtered = filter_evidence(
        claim,
        evidence,
        min_overlap=0.20,
    )

    ranked = rank_evidence(
        claim,
        filtered,
    )

    nli_results = []

    for item in ranked:

        text = " ".join([
            item.get("title", ""),
            item.get("content", ""),
        ])

        nli = verify_claim(
            claim,
            text,
        )

        item = item.copy()
        item.update(nli)

        nli_results.append(item)

    aggregated = aggregate_evidence(
        nli_results
    )

    return decide_verdict(
        aggregated
    )


def test_supported_claim():

    claim = "India launched a new AI policy in 2026."

    evidence = [
        {
            "title": "India launches new AI policy",
            "content": (
                "India launched a new AI policy in 2026 "
                "to support artificial intelligence development."
            ),
            "url": "https://example.gov.in/ai-policy",
            "published_date": "2026-02-10",
        }
    ]

    result = run_pipeline(claim, evidence)

    assert result["verdict"] == "REAL"


def test_refuted_claim():

    claim = "India launched a new AI policy in 2026."

    evidence = [
        {
            "title": "India did not launch AI policy",
            "content": (
                "India did not launch a new AI policy in 2026."
            ),
            "url": "https://example.gov.in/fact-check",
            "published_date": "2026-02-10",
        }
    ]

    result = run_pipeline(claim, evidence)

    assert result["verdict"] == "FAKE"


def test_unrelated_claim():

    claim = "India launched a new AI policy in 2026."

    evidence = [
        {
            "title": "Michelin announces new tyres",
            "content": (
                "Michelin announced a new range of tyres "
                "for passenger cars."
            ),
            "url": "https://example.com/tyres",
            "published_date": "2026-02-10",
        }
    ]

    result = run_pipeline(claim, evidence)

    assert result["verdict"] == "UNVERIFIED"