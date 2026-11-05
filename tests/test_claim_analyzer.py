from src.nlp.claim_analyzer import analyze_claim


def test_claim_analysis():
    claim = "India announced a new artificial intelligence policy in 2026."

    result = analyze_claim(claim)

    assert result["claim"] == claim
    assert result["claim_type"] == "announcement"
    assert "India" in result["entities"]
    assert "2026" in result["dates"]


def test_empty_claim():
    try:
        analyze_claim("")
        assert False
    except ValueError:
        assert True