"""
search.py: Implements web search functionality for the Chat99 system.
"""

import os
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load your API key and Custom Search Engine ID from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")


def google_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    if not API_KEY or not SEARCH_ENGINE_ID:
        logger.error("Google API key or Search Engine ID is missing")
        return []
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = (
            service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=num_results).execute()
        )

        results = []
        for item in res.get("items", []):
            results.append(
                {
                    "title": item["title"],
                    "link": item["link"],
                    "snippet": item["snippet"],
                }
            )

        return results
    except HttpError as e:
        logger.error(f"An error occurred during the Google search: {str(e)}")
        return []


def perform_search(query: str) -> Optional[str]:
    """
    Perform a web search and return the results as a formatted string.

    Args:
        query (str): The search query.

    Returns:
        Optional[str]: A formatted string of search results, or None if the search failed.
    """
    search_results = google_search(query)
    if not search_results:
        return None

    formatted_results = "Search Results:\n"
    for i, result in enumerate(search_results, 1):
        formatted_results += (
            f"{i}. {result['title']}\n   {result['link']}\n   {result['snippet']}\n\n"
        )

    return formatted_results
