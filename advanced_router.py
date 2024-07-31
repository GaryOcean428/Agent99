"""
AdvancedRouter: Dynamically selects the best model and parameters for a given query or task.
"""

from typing import Dict, Any
import re
from models import get_model_list
from config import ROUTER_THRESHOLD, HIGH_TIER_MODEL, MID_TIER_MODEL, LOW_TIER_MODEL

class AdvancedRouter:
    def __init__(self):
        self.models = get_model_list()
        self.local_model = "llama3.1:8b"
        self.threshold = ROUTER_THRESHOLD

    def route(self, query: str, conversation_history: list) -> Dict[str, Any]:
        """
        Determine the best model and parameters for the given query and conversation history.
        """
        complexity = self._assess_complexity(query)
        context_length = self._calculate_context_length(conversation_history)
        task_type = self._identify_task_type(query)

        if complexity < self.threshold / 2 and context_length < 1000:
            config = self._get_low_tier_config(task_type)
        elif complexity < self.threshold and context_length < 4000:
            config = self._get_mid_tier_config(task_type)
        else:
            config = self._get_high_tier_config(task_type)

        config['routing_explanation'] = f"Selected {config['model']} based on complexity ({complexity:.2f}) and context length ({context_length} chars). Threshold: {self.threshold}"
        return config

    def determine_complexity(self, query: str) -> str:
        """Determine the complexity of the query on a scale of high, mid, low."""
        complexity_score = self._assess_complexity(query)
        if complexity_score < 0.3:
            return "low"
        elif complexity_score < 0.7:
            return "mid"
        else:
            return "high"

    def _assess_complexity(self, query: str) -> float:
        """
        Assess the complexity of the query on a scale of 0 to 1.
        """
        # Implement complexity assessment logic here
        # This is a placeholder implementation
        return len(query) / 1000  # Simplified complexity measure

    def _calculate_context_length(self, conversation_history: list) -> int:
        """
        Calculate the total length of the conversation history.
        """
        return sum(len(message['content']) for message in conversation_history)

    def _identify_task_type(self, query: str) -> str:
        """
        Identify the type of task based on the query.
        """
        # Implement task type identification logic here
        # This is a placeholder implementation
        return "general"

    def _get_low_tier_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for low-tier model processing.
        """
        return {
            'model': LOW_TIER_MODEL,
            'max_tokens': 256,
            'temperature': 0.5,
        }

    def _get_mid_tier_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for mid-tier model processing.
        """
        return {
            'model': MID_TIER_MODEL,
            'max_tokens': 512,
            'temperature': 0.7,
        }

    def _get_high_tier_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for high-tier model processing.
        """
        return {
            'model': HIGH_TIER_MODEL,
            'max_tokens': 1024,
            'temperature': 0.9,
        }

advanced_router = AdvancedRouter()