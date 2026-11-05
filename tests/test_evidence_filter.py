from src.evidence.evidence_filter import filter_evidence


def test_irrelevant_evidence_is_removed():

    claim = (
        "India announced a new artificial intelligence policy "
        "in 2026."
    )

    results = [
        {
            "title": "India announces AI policy",
            "content": (
                "India announced a new artificial intelligence "
                "policy in 2026."
            ),
            "url": "https://example.com/ai",
        },
        {
            "title": "Michelin launches new tyres",
            "content": (
                "Michelin launched new tyres for cars."
            ),
            "url": "https://example.com/tyres",
        },
    ]

    filtered = filter_evidence(
        claim,
        results
    )

    urls = [item["url"] for item in filtered]

    assert "https://example.com/ai" in urls
    assert "https://example.com/tyres" not in urls


def test_empty_results():

    result = filter_evidence(
        "India announced an AI policy.",
        []
    )

    assert result == []