from src.explainability.xai_explainer import explain_prediction


def test_xai_returns_prediction():

    text = (
        "The government announced a new technology policy today."
    )

    result = explain_prediction(
        text,
        num_features=5,
    )

    assert result["prediction"] in ["FAKE", "REAL"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["features"]) > 0


def test_xai_rejects_empty_text():

    try:
        explain_prediction("")
        assert False
    except ValueError:
        assert True