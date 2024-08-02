import os
from googleapiclient.discovery import build
from serpapi.google_search import GoogleSearch


def perform_search(query: str) -> str:
    google_results = google_search(query)
    serp_results = serp_search(query)

    combined_results = (
        f"Google Results:\n{google_results}\n\nSERP Results:\n{serp_results}"
    )
    return combined_results
    google_results = google_search(query)
    serp_results = serp_search(query)

    combined_results = (
        f"Google Results:\n{google_results}\n\nSERP Results:\n{serp_results}"
    )
    return combined_results


def google_search(query: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=5).execute()

    results = []
    for item in res.get("items", []):
        results.append(
            f"Title: {item['title']}\nSnippet: {item['snippet']}\nLink: {item['link']}\n"
        )

    return "\n".join(results)


def serp_search(query: str) -> str:
    api_key = os.getenv("SERP_API_KEY")
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": 5,
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic_results = results.get("organic_results", [])
    formatted_results = []
    for result in organic_results:
        formatted_results.append(
            f"Title: {result['title']}\nSnippet: {result['snippet']}\nLink: {result['link']}\n"
        )

    return "\n".join(formatted_results)
