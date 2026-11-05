import os

import requests
from dotenv import load_dotenv


load_dotenv()


def search_web(query, max_results=5):
    """
    Search the web using a configurable search API.

    The function returns a consistent article structure
    regardless of the search provider.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured."
        )

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for item in data.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "source": item.get("url", ""),
            }
        )

    return results


if __name__ == "__main__":

    query = "India artificial intelligence policy 2026"

    results = search_web(query, max_results=5)

    print("=" * 50)
    print("WEB SEARCH TEST")
    print("=" * 50)

    print(f"Results: {len(results)}")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(result["url"])