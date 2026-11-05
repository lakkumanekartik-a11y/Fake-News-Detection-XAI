from src.decision.decision_engine import decide_verdict


def test_real_verdict():

    aggregated = {
        "support_score": 0.90,
        "refute_score": 0.10,
        "support_count": 2,
        "refute_count": 1,
    }

    result = decide_verdict(aggregated)

    assert result["verdict"] == "REAL"
    assert result["confidence"] >= 0.70


def test_fake_verdict():

    aggregated = {
        "support_score": 0.10,
        "refute_score": 0.90,
        "support_count": 1,
        "refute_count": 2,
    }

    result = decide_verdict(aggregated)

    assert result["verdict"] == "FAKE"
    assert result["confidence"] >= 0.70


def test_unverified_with_insufficient_evidence():

    aggregated = {
        "support_score": 0.30,
        "refute_score": 0.20,
        "support_count": 1,
        "refute_count": 1,
    }

    result = decide_verdict(aggregated)

    assert result["verdict"] == "UNVERIFIED"


def test_unverified_with_conflicting_evidence():

    aggregated = {
        "support_score": 0.85,
        "refute_score": 0.90,
        "support_count": 1,
        "refute_count": 1,
    }

    result = decide_verdict(aggregated)

    assert result["verdict"] == "FAKE"