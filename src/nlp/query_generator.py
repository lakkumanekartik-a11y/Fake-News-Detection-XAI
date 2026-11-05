from src.nlp.claim_analyzer import analyze_claim


def generate_queries(claim):
    analysis = analyze_claim(claim)

    clean_claim = analysis["claim"]
    entities = analysis["entities"]
    dates = analysis["dates"]

    queries = []

    # Exact claim
    queries.append(f'"{clean_claim}"')

    # Entity + claim
    if entities:
        entity_text = " ".join(entities)
        queries.append(f"{entity_text} {clean_claim}")

    # Entity + date + claim type
    if entities and dates:
        entity_text = " ".join(entities)
        date_text = " ".join(dates)
        queries.append(
            f"{entity_text} {date_text} {analysis['claim_type']}"
        )

    # Official government search
    if entities:
        entity_text = " ".join(entities)
        queries.append(
            f"site:gov.in {entity_text} {analysis['claim_type']}"
        )

    # Remove duplicates
    queries = [" ".join(query.split()) for query in queries]
    queries = list(dict.fromkeys(queries))

    return queries