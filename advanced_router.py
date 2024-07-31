"""
AdvancedRouter: Dynamically selects the best model, parameters, and response strategy for a given query or task.
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
        Determine the best model, parameters, and response strategy for the given query and conversation history.
        """
        complexity = self._assess_complexity(query)
        context_length = self._calculate_context_length(conversation_history)
        task_type = self._identify_task_type(query)
        question_type = self._classify_question(query)

        if complexity < self.threshold / 2 and context_length < 1000:
            config = self._get_low_tier_config(task_type)
        elif complexity < self.threshold and context_length < 4000:
            config = self._get_mid_tier_config(task_type)
        else:
            config = self._get_high_tier_config(task_type)

        config['routing_explanation'] = f"Selected {config['model']} based on complexity ({complexity:.2f}) and context length ({context_length} chars). Threshold: {self.threshold}"
        config['question_type'] = question_type
        config['response_strategy'] = self._get_response_strategy(question_type)

        return config

    def _assess_complexity(self, query: str) -> float:
        """
        Assess the complexity of the query on a scale of 0 to 1.
        """
        # Implement more sophisticated complexity assessment logic here
        word_count = len(query.split())
        sentence_count = len(re.findall(r'\w+[.!?]', query)) + 1
        avg_word_length = sum(len(word) for word in query.split()) / word_count if word_count > 0 else 0
        
        complexity = (word_count / 100) * 0.4 + (sentence_count / 10) * 0.3 + (avg_word_length / 10) * 0.3
        return min(complexity, 1.0)  # Ensure complexity is between 0 and 1

    def _calculate_context_length(self, conversation_history: list) -> int:
        """
        Calculate the total length of the conversation history.
        """
        return sum(len(message['content']) for message in conversation_history)

    def _identify_task_type(self, query: str) -> str:
        """
        Identify the type of task based on the query.
        """
        query_lower = query.lower()
        if any(word in query_lower for word in ['code', 'program', 'function', 'debug']):
            return "coding"
        elif any(word in query_lower for word in ['analyze', 'compare', 'evaluate']):
            return "analysis"
        elif any(word in query_lower for word in ['create', 'generate', 'write']):
            return "creative"
        else:
            return "general"

    def _classify_question(self, query: str) -> str:
        """
        Classify the type of question.
        """
        query_lower = query.lower()
        if any(word in query_lower for word in ['how', 'why', 'explain']):
            return "problem_solving"
        elif any(word in query_lower for word in ['what', 'who', 'where', 'when']):
            return "factual"
        elif query_lower.startswith(('is', 'are', 'can', 'do', 'does')):
            return "yes_no"
        elif any(word in query_lower for word in ['compare', 'contrast', 'analyze']):
            return "analysis"
        else:
            return "open_ended"

    def _get_response_strategy(self, question_type: str) -> str:
        """
        Determine the appropriate response strategy based on question type.
        """
        strategy_map = {
            "problem_solving": "irac",
            "factual": "direct_answer",
            "yes_no": "boolean_with_explanation",
            "analysis": "comparative_analysis",
            "open_ended": "open_discussion"
        }
        return strategy_map.get(question_type, "default")

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