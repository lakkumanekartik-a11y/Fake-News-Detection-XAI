from src.evidence.nli_verifier import verify_claim


def test_supports_claim():

    claim = "India launched a new AI policy in 2026."

    evidence = (
        "India launched a new AI policy in 2026."
    )

    result = verify_claim(
        claim,
        evidence,
    )

    assert result["label"] == "SUPPORTS"
    assert result["confidence"] >= 0.70


def test_refutes_claim():

    claim = "India launched a new AI policy in 2026."

    evidence = (
        "India did not launch a new AI policy in 2026."
    )

    result = verify_claim(
        claim,
        evidence,
    )

    assert result["label"] == "REFUTES"
    assert result["confidence"] >= 0.70


def test_unrelated_evidence():

    claim = "India launched a new AI policy in 2026."

    evidence = (
        "Michelin announced a new range of tyres for cars."
    )

    result = verify_claim(
        claim,
        evidence,
    )

    assert result["label"] == "NEUTRAL"
    assert result["confidence"] >= 0.70