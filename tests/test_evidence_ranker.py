from src.evidence.evidence_ranker import rank_evidence


def test_relevant_official_source_ranks_first():

    claim = (
        "India announced a new artificial intelligence policy "
        "in 2026."
    )

    results = [
        {
            "title": "Generic Technology News",
            "content": (
                "A company announced a new technology product."
            ),
            "url": "https://example.com/news",
        },
        {
            "title": "India AI Policy",
            "content": (
                "India announced a new artificial intelligence "
                "policy in 2026."
            ),
            "url": "https://www.pib.gov.in/example",
        },
    ]

    ranked = rank_evidence(
        claim,
        results,
    )

    assert ranked[0]["url"] == (
        "https://www.pib.gov.in/example"
    )


def test_scores_are_present():

    claim = "India announced an AI policy."

    results = [
        {
            "title": "India AI policy",
            "content": "India announced an AI policy.",
            "url": "https://www.pib.gov.in/example",
        }
    ]

    ranked = rank_evidence(
        claim,
        results,
    )

    assert "relevance" in ranked[0]
    assert "entity_match" in ranked[0]
    assert "source_quality" in ranked[0]
    assert "evidence_score" in ranked[0]

def test_current_evidence_gets_higher_date_score():

    claim = (
        "India announced a new artificial intelligence "
        "policy in 2026."
    )

    results = [
        {
            "title": "India AI Policy 2020",
            "content": (
                "India discussed artificial intelligence "
                "policy in 2020."
            ),
            "url": "https://example.com/old",
            "published_date": "2020-05-10",
        },
        {
            "title": "India AI Policy 2026",
            "content": (
                "India announced a new artificial intelligence "
                "policy in 2026."
            ),
            "url": "https://www.pib.gov.in/example",
            "published_date": "2026-02-10",
        },
    ]

    ranked = rank_evidence(
        claim,
        results,
    )

    assert ranked[0]["published_date"] == "2026-02-10"
    assert ranked[0]["date_relevance"] == 1.0