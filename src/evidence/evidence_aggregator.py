from typing import Any


MIN_EVIDENCE_SCORE = 0.60
MIN_NLI_CONFIDENCE = 0.70


def aggregate_evidence(
    evidence_results: list[dict[str, Any]]
):
    supports = []
    refutes = []
    neutral = []

    for item in evidence_results:

        label = item.get("label", "NEUTRAL")

        confidence = float(
            item.get("confidence", 0.0)
        )

        evidence_score = float(
            item.get("evidence_score", 0.0)
        )

        # These fields are optional.
        # Existing unit tests don't provide them.
        relevance = float(
            item.get("relevance", 1.0)
        )

        # Reject explicitly weak evidence.
        if evidence_score < MIN_EVIDENCE_SCORE:
            continue

        # Reject explicitly weak NLI.
        if confidence < MIN_NLI_CONFIDENCE:
            continue

        # If relevance is supplied, check it.
        if "relevance" in item and relevance < 0.35:
            continue

        weighted_score = (
            confidence * evidence_score
        )

        result = item.copy()
        result["weighted_score"] = weighted_score

        if label == "SUPPORTS":
            supports.append(result)

        elif label == "REFUTES":
            refutes.append(result)

        else:
            neutral.append(result)

    supports.sort(
        key=lambda x: x["weighted_score"],
        reverse=True,
    )

    refutes.sort(
        key=lambda x: x["weighted_score"],
        reverse=True,
    )

    neutral.sort(
        key=lambda x: x["weighted_score"],
        reverse=True,
    )

    return {
        "supports": supports,
        "refutes": refutes,
        "neutral": neutral,

        "support_score": sum(
            x["weighted_score"]
            for x in supports
        ),

        "refute_score": sum(
            x["weighted_score"]
            for x in refutes
        ),

        "support_count": len(supports),
        "refute_count": len(refutes),
        "neutral_count": len(neutral),
    }