from src.evidence.evidence_aggregator import aggregate_evidence


def test_supporting_evidence_is_aggregated():

    evidence = [
        {
            "label": "SUPPORTS",
            "confidence": 0.95,
            "evidence_score": 0.90,
        },
        {
            "label": "SUPPORTS",
            "confidence": 0.85,
            "evidence_score": 0.80,
        },
    ]

    result = aggregate_evidence(evidence)

    assert result["support_count"] == 2
    assert result["refute_count"] == 0
    assert result["neutral_count"] == 0
    assert result["support_score"] > 0


def test_refuting_evidence_is_aggregated():

    evidence = [
        {
            "label": "REFUTES",
            "confidence": 0.95,
            "evidence_score": 0.90,
        },
        {
            "label": "REFUTES",
            "confidence": 0.80,
            "evidence_score": 0.85,
        },
    ]

    result = aggregate_evidence(evidence)

    assert result["support_count"] == 0
    assert result["refute_count"] == 2
    assert result["neutral_count"] == 0
    assert result["refute_score"] > 0


def test_mixed_evidence_is_separated():

    evidence = [
        {
            "label": "SUPPORTS",
            "confidence": 0.90,
            "evidence_score": 0.90,
        },
        {
            "label": "REFUTES",
            "confidence": 0.80,
            "evidence_score": 0.80,
        },
        {
            "label": "NEUTRAL",
            "confidence": 0.90,
            "evidence_score": 0.70,
        },
    ]

    result = aggregate_evidence(evidence)

    assert result["support_count"] == 1
    assert result["refute_count"] == 1
    assert result["neutral_count"] == 1