"""
Custom configuration for RouteLLM to use our OpenAI client.
"""

import os
from openai import OpenAI

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it and try again.")

# Initialize OpenAI client
OPENAI_CLIENT = OpenAI(api_key=openai_api_key)

# Override the RouteLLM's OPENAI_CLIENT
import routellm.routers.similarity_weighted.utils
routellm.routers.similarity_weighted.utils.OPENAI_CLIENT = OPENAI_CLIENT