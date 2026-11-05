from src.retrieval.retriever import retrieve_evidence
from src.evidence.evidence_filter import filter_evidence
from src.evidence.evidence_ranker import rank_evidence
from src.evidence.nli_verifier import verify_claim
from src.evidence.evidence_aggregator import aggregate_evidence
from src.decision.decision_engine import decide_verdict


claim = "RCB lost the Indian Premier League final in 2011."

print("\n" + "=" * 70)
print("RCB LIVE PIPELINE DEBUG")
print("=" * 70)

# 1. Retrieval
retrieved = retrieve_evidence(claim)

print("\nRETRIEVED:", len(retrieved["results"]))

# 2. Filter
filtered = filter_evidence(
    claim,
    retrieved["results"]
)

print("FILTERED:", len(filtered))

# 3. Rank
ranked = rank_evidence(
    claim,
    filtered
)

print("RANKED:", len(ranked))

# 4. NLI
verified = []

print("\n" + "-" * 70)
print("EVIDENCE + NLI")
print("-" * 70)

for i, item in enumerate(ranked[:5], 1):

    evidence_text = (
        item.get("title", "")
        + " "
        + item.get("content", "")
    )

    nli = verify_claim(
        claim,
        evidence_text
    )

    item = item.copy()
    item.update(nli)

    verified.append(item)

    print(f"\n[{i}]")
    print("TITLE:", item.get("title"))
    print("URL:", item.get("url"))
    print("Evidence Score:", item.get("evidence_score"))
    print("Relevance:", item.get("relevance"))
    print("NLI:", nli["label"])
    print("NLI Confidence:", nli["confidence"])
    print("NLI Scores:", nli["scores"])
    print("Relevant Sentences:")

    for sentence in nli["relevant_sentences"]:
        print("  -", sentence[:500])

# 5. Aggregation
aggregated = aggregate_evidence(verified)

print("\n" + "=" * 70)
print("AGGREGATION")
print("=" * 70)

print("Support count:", aggregated["support_count"])
print("Refute count:", aggregated["refute_count"])
print("Support score:", aggregated["support_score"])
print("Refute score:", aggregated["refute_score"])

# 6. Decision
decision = decide_verdict(aggregated)

print("\n" + "=" * 70)
print("FINAL DECISION")
print("=" * 70)

print("VERDICT:", decision["verdict"])
print("CONFIDENCE:", decision["confidence"])
print("REASON:", decision["reason"])