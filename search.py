"""
This module provides functionality for performing web searches using Google Custom Search API
and SERP API. It combines results from both sources to provide comprehensive search results.
"""

import os
import logging

logger = logging.getLogger(__name__)

try:
    from googleapiclient.discovery import build

    logger.info("Successfully imported googleapiclient.discovery")
except ImportError as e:
    logger.error("Failed to import googleapiclient.discovery: %s", str(e))
    logger.info("Python path: %s", os.sys.path)

try:
    from serpapi import GoogleSearch

    logger.info("Successfully imported serpapi.GoogleSearch")
except ImportError as e:
    logger.error("Failed to import serpapi.GoogleSearch: %s", str(e))


def perform_search(query: str) -> str:
    """
    Perform a web search using both Google Custom Search and SERP API.

    Args:
        query (str): The search query string.

    Returns:
        str: A combined string of search results from both APIs.
    """
    google_results = google_search(query)
    serp_results = serp_search(query)

    combined_results = (
        f"Google Results:\n{google_results}\n\nSERP Results:\n{serp_results}"
    )
    return combined_results


def google_search_custom(query: str) -> str:
    """
    Perform a search using Google Custom Search API.

    Args:
        query (str): The search query string.

    Returns:
        str: A formatted string of search results.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        logger.error("Google API key or Custom Search Engine ID is missing")
        return "Google search is currently unavailable."

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=5).execute()

        results = []
        for item in res.get("items", []):
            results.append(
                f"Title: {item['title']}\nSnippet: {item['snippet']}\nLink: {item['link']}\n"
            )

        return "\n".join(results)
    except ImportError as error:
        logger.error("Error in Google search: %s", str(error))
        return "An error occurred while performing the Google search."


def google_search(query: str) -> str:
    """
    Perform a Google search using the given query.

    Args:
        query (str): The search query.

    Returns:
        str: The search results as a formatted string.

    Raises:
        Exception: If an error occurs during the search.

    """
    # Function implementation...
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        logger.error("Google API key or Custom Search Engine ID is missing or invalid")
        return "Google search is currently unavailable."

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=5).execute()

        results = []
        for item in res.get("items", []):
            results.append(
                f"Title: {item['title']}\nSnippet: {item['snippet']}\nLink: {item['link']}\n"
            )

        return "\n".join(results)
    except ImportError as inner_exception:
        logger.error("Error in Google search: %s", str(inner_exception))
        return "An error occurred while performing the Google search."


def serp_search(query: str) -> str:
    """
    Perform a search using SERP API.

    Args:
        query (str): The search query string.

    Returns:
        str: A formatted string of search results.
    """
    serp_api_key = os.getenv("SERP_API_KEY")

    if not serp_api_key:
        logger.error("SERP API key is missing")
        return "SERP search is currently unavailable."

    params = {
        "engine": "google",
        "q": query,
        "api_key": serp_api_key,
        "num": 5,
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        organic_results = results.get("organic_results", [])
        formatted_results = []
        for result in organic_results:
            formatted_results.append(
                f"Title: {result['title']}\nSnippet: {result['snippet']}\nLink: {result['link']}\n"
            )

        return "\n".join(formatted_results)
    except ImportError as inner_exception:
        logger.error("Error in SERP search: %s", str(inner_exception))
        return "An error occurred while performing the SERP search."
