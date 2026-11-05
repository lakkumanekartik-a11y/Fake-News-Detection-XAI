from src.nlp.query_generator import generate_queries


def test_query_generation():
    claim = "India announced a new artificial intelligence policy in 2026."

    queries = generate_queries(claim)

    assert len(queries) >= 3

    combined = " ".join(queries).lower()

    assert "india" in combined
    assert "2026" in combined
    assert "artificial intelligence" in combined


def test_queries_are_unique():
    claim = "India announced a new artificial intelligence policy in 2026."

    queries = generate_queries(claim)

    assert len(queries) == len(set(queries))