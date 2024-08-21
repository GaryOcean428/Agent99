"""
This module handles the generation of responses using various AI models.
"""

import os
import logging
from typing import List, Dict
from anthropic import Anthropic, APIError, APIConnectionError, AuthenticationError, RateLimitError
from groq import Groq
from input_analyzer import analyze_input

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize API clients
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class ModelManager:
    """Manages different AI models and handles response generation."""

    def __init__(self):
        self.models = {
            "low": "llama-3.1-8b-instant",
            "mid": "llama-3.1-70b-versatile",
            "high": "llama-3.1-405b-instruct",
            "superior": "claude-3-5-sonnet-20240620",
        }
        self.thought_process = """
        1. Understand the question or task
        2. Break down the problem into smaller parts
        3. Consider relevant information and context
        4. Analyze potential approaches or solutions
        5. Draw conclusions or provide a step-by-step explanation
        6. Summarize the response
        """

    def _prepare_prompt(self, system_prompt: str, input_type: str) -> str:
        """Prepare the system prompt based on input type."""
        if input_type == "complex":
            return f"{system_prompt}\n\nFor complex queries, follow this thought process:\n{self.thought_process}"
        return system_prompt

    def _call_anthropic_api(
        self,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call the Anthropic API and handle potential errors."""
        try:
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text
        except (APIError, APIConnectionError, AuthenticationError, RateLimitError) as e:
            logger.error("Anthropic API error: %s", str(e))
        except Exception as e:
            logger.error("Unexpected error with Anthropic API: %s", str(e))
        return ""

    def _call_groq_api(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call the Groq API and handle potential errors."""
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error with Groq API: %s", str(e))
        return ""

    def generate_response(
        self,
        model_tier: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Generate a response using the appropriate API based on the model tier.

        Args:
            model_tier (str): The tier of the model to use (low, mid, high, superior).
            system_prompt (str): The system prompt to guide the AI's behavior.
            messages (List[Dict[str, str]]): The conversation history.
            max_tokens (int): The maximum number of tokens in the response.
            temperature (float): The randomness of the response.

        Returns:
            str: The generated response from the AI model.
        """
        model = self.models.get(model_tier)
        if not model:
            logger.error("Invalid model tier: %s", model_tier)
            return ""

        user_input = messages[-1]["content"]
        input_type = analyze_input(user_input)
        prepared_prompt = self._prepare_prompt(system_prompt, input_type)

        if model_tier == "superior":
            return self._call_anthropic_api(
                model, prepared_prompt, messages, max_tokens, temperature
            )
        else:
            return self._call_groq_api(model, messages, max_tokens, temperature)


class ResponseGenerator:
    """Generates responses using the ModelManager."""

    def __init__(self, config, memory_manager):
        self.config = config
        self.memory_manager = memory_manager
        self.model_manager = ModelManager()

    def generate(self, analyzed_input):
        """
        Generate a response based on the analyzed input.

        Args:
            analyzed_input (str): The analyzed user input.

        Returns:
            str: The generated response.
        """
        system_prompt = "You are a helpful AI assistant."
        messages = [{"role": "user", "content": analyzed_input}]
        response = self.model_manager.generate_response(
            model_tier="mid",
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=100,
            temperature=0.7,
        )
        return response if response else "I'm sorry, I couldn't generate a response at this time."


model_manager = ModelManager()
