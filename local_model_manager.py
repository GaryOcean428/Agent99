"""
LocalModelManager: Handles interactions with local language models.
"""

from typing import List, Dict
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LOCAL_MODEL_PATH

class LocalModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._load_models()

    def _load_models(self):
        """Load local models from the specified directory."""
        for model_name in os.listdir(LOCAL_MODEL_PATH):
            model_path = os.path.join(LOCAL_MODEL_