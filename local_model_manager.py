"""
LocalModelManager: Handles interactions with the local Llama model using Ollama.
"""

import requests

class LocalModelManager:
    def __init__(self, model_name="llama3.1:8b", api_base="http://localhost:11434"):
        self.model_name = model_name
        self.api_base = api_base

    def generate_response(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate a response using the local Llama model."""
        try:
            response = requests.post(
                f"{self.api_base}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.RequestException as e:
            print(f"Error communicating with Ollama: {str(e)}")
            return None

    def is_available(self) -> bool:
        """Check if the local model is available."""
        try:
            response = requests.get(f"{self.api_base}/api/tags")
            return response.status_code == 200
        except requests.RequestException:
            return False

local_model_manager = LocalModelManager()
