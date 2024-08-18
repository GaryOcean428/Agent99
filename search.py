import os
import logging
from typing import Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load your API key and Custom Search Engine ID from environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv(
    "GOOGLE_CSE_ID"
)  # Ensure this matches your environment variable


def perform_search(query: str, num_results: int = 5) -> str:
    """
    Perform a Google search using the Custom Search API.

    Args:
        query (str): The search query.
        num_results (int): The number of search results to return. Default is 5.

    Returns:
        str: A formatted string of search results, or an error message.
    """
    if not API_KEY or not SEARCH_ENGINE_ID:
        logger.error("Google API key or Search Engine ID is missing")
        return "Error: Missing API key or Search Engine ID."

    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = (
            service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=num_results).execute()
        )

        search_results = []
        for item in res.get("items", []):
            search_results.append(
                f"Title: {item['title']}\nLink: {item['link']}\nSnippet: {item['snippet']}\n"
            )

        return (
            "\n".join(search_results) if search_results else "No search results found."
        )
    except HttpError as e:
        logger.error(f"An error occurred during the Google search: {str(e)}")
        return "Error performing search."


if __name__ == "__main__":
    # Example usage
    test_query = "Artificial Intelligence latest developments"
    results = perform_search(test_query)
    if results:
        print(results)
    else:
        print("No results found or an error occurred.")
