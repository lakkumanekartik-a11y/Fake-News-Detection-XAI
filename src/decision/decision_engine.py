from typing import Any


MIN_SCORE = 0.70


def decide_verdict(
    aggregated: dict[str, Any]
):

    support_score = float(
        aggregated.get(
            "support_score",
            0.0
        )
    )

    refute_score = float(
        aggregated.get(
            "refute_score",
            0.0
        )
    )

    support_count = int(
        aggregated.get(
            "support_count",
            0
        )
    )

    refute_count = int(
        aggregated.get(
            "refute_count",
            0
        )
    )

    # --------------------------------------------------
    # Strong support
    # --------------------------------------------------

    if (
        support_score >= MIN_SCORE
        and support_count >= 1
        and support_score > refute_score
    ):

        return {
            "verdict": "REAL",
            "confidence": min(
                support_score,
                0.99
            ),
            "reason": (
                "The available evidence supports "
                "the claim."
            ),
        }

    # --------------------------------------------------
    # Strong refutation
    # --------------------------------------------------

    if (
        refute_score >= MIN_SCORE
        and refute_count >= 1
        and refute_score > support_score
    ):

        return {
            "verdict": "FAKE",
            "confidence": min(
                refute_score,
                0.99
            ),
            "reason": (
                "The available evidence contradicts "
                "the claim."
            ),
        }

    # --------------------------------------------------
    # Not enough evidence
    # --------------------------------------------------

    return {
        "verdict": "UNVERIFIED",
        "confidence": 0.0,
        "reason": (
            "There is not enough consistent evidence "
            "to verify the claim."
        ),
    }