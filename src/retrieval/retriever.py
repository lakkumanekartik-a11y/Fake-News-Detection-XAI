from src.nlp.query_generator import generate_queries
from src.retrieval.web_search import search_web


def deduplicate_results(results):
    seen_urls = set()
    unique_results = []

    for result in results:
        url = result.get("url", "").strip()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)
        unique_results.append(result)

    return unique_results


def retrieve_evidence(claim, max_results_per_query=5):
    queries = generate_queries(claim)

    all_results = []

    for query in queries:
        try:
            results = search_web(
                query,
                max_results=max_results_per_query,
            )

            all_results.extend(results)

        except Exception as exc:
            print(f"[WARNING] Search failed: {exc}")

    unique_results = deduplicate_results(all_results)

    return {
        "claim": claim,
        "queries": queries,
        "results": unique_results,
    }


if __name__ == "__main__":

    claim = (
        "India announced a new artificial intelligence policy "
        "in 2026."
    )

    data = retrieve_evidence(claim)

    print("=" * 60)
    print("RETRIEVAL PIPELINE")
    print("=" * 60)

    print(f"Queries generated : {len(data['queries'])}")
    print(f"Unique results    : {len(data['results'])}")

    print("\nTOP RESULTS:")

    for i, result in enumerate(data["results"][:10], 1):
        print(f"\n{i}. {result['title']}")
        print(result["url"])