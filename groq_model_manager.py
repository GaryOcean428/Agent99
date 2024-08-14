"""
GroqModelManager: Handles interactions with the Groq API for faster inference with multiple models.
"""

import os
from groq import Groq
from typing import List, Dict


class GroqModelManager:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.models = {
            "low": "llama-3.1-8b-instant",
            "mid": "llama-3.1-70b-versatile",
            "high": "llama-3.1-405b-instruct"  # Not yet available, but prepared for future use
        }

    def generate_response(self, messages: List[Dict[str, str]], model_tier: str = "mid", max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate a response using the Groq API with the specified model tier."""
        model = self.models.get(model_tier, self.models["mid"])
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response from Groq ({model}): {str(e)}")
            return ""

    def is_available(self) -> bool:
        """Check if the Groq API is available."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


groq_model_manager = GroqModelManager()
