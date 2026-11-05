from src.retrieval.retriever import deduplicate_results


def test_deduplicate_results():
    results = [
        {
            "title": "Article 1",
            "url": "https://example.com/a",
            "content": "Text",
            "source": "https://example.com/a",
        },
        {
            "title": "Article 1 duplicate",
            "url": "https://example.com/a",
            "content": "Different text",
            "source": "https://example.com/a",
        },
        {
            "title": "Article 2",
            "url": "https://example.com/b",
            "content": "Text",
            "source": "https://example.com/b",
        },
    ]

    unique = deduplicate_results(results)

    assert len(unique) == 2
    assert unique[0]["url"] == "https://example.com/a"
    assert unique[1]["url"] == "https://example.com/b"


def test_empty_results():
    assert deduplicate_results([]) == []