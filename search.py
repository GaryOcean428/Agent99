import os
import logging
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load your API key and Custom Search Engine ID from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")


def google_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a Google search using the Custom Search API.

    Args:
        query (str): The search query.
        num_results (int): The number of search results to return. Default is 5.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing search results.
    """
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

        logger.info(
            f"Successfully retrieved {len(results)} search results for query: {query}"
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
        logger.warning(f"No search results found for query: {query}")
        return None

    formatted_results = "Search Results:\n"
    for i, result in enumerate(search_results, 1):
        formatted_results += (
            f"{i}. {result['title']}\n   {result['link']}\n   {result['snippet']}\n\n"
        )

    logger.info(f"Formatted search results for query: {query}")
    return formatted_results


if __name__ == "__main__":
    # Example usage
    test_query = "Artificial Intelligence latest developments"
    results = perform_search(test_query)
    if results:
        print(results)
    else:
        print("No results found or an error occurred.")
