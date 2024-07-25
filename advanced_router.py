"""
AdvancedRouter: Dynamically selects the best model and parameters for a given query or task.
"""

from typing import Dict, Any
import re
from models import get_model_list

class AdvancedRouter:
    def __init__(self):
        self.models = get_model_list()
        self.local_model = "llama3.1:8b"

    def route(self, query: str, conversation_history: list) -> Dict[str, Any]:
        """
        Determine the best model and parameters for the given query and conversation history.
        """
        complexity = self._assess_complexity(query)
        context_length = self._calculate_context_length(conversation_history)
        task_type = self._identify_task_type(query)

        if complexity < 0.1 and context_length < 100:
            config = self._get_local_config(task_type)
            config['routing_explanation'] = f"Using local model due to very low complexity ({complexity:.2f}) and short context ({context_length} chars)."
            return config
        elif complexity < 0.3 and context_length < 1000:
            config = self._get_low_tier_config(task_type)
            config['routing_explanation'] = f"Using low-tier model due to low complexity ({complexity:.2f}) and short context ({context_length} chars)."
            return config
        elif complexity < 0.7 and context_length < 4000:
            config = self._get_mid_tier_config(task_type)
            config['routing_explanation'] = f"Using mid-tier model due to moderate complexity ({complexity:.2f}) and medium context ({context_length} chars)."
            return config
        else:
            config = self._get_high_tier_config(task_type)
            config['routing_explanation'] = f"Using high-tier model due to high complexity ({complexity:.2f}) or long context ({context_length} chars)."
            return config

    def _assess_complexity(self, query: str) -> float:
        """
        Assess the complexity of the query on a scale of 0 to 1.
        """
        # Improved heuristics for complexity assessment
        complexity_indicators = [
            r'\b(explain|analyze|compare|contrast|evaluate|synthesize)\b',
            r'\b(code|program|algorithm|function)\b',
            r'\b(scientific|technical|academic)\b',
            r'\b(why|how|what if)\b',
            r'\b(\w+\s+){5,}',  # Checks if query has more than 5 words
        ]
        simple_indicators = [
            r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$',  # Simple arithmetic
            r'^\s*what\s+is\s+\d+\s*[\+\-\*/]\s*\d+\s*$',  # "What is" followed by simple arithmetic
            r'^\s*hi|hello|hey\s*$',  # Simple greetings
        ]
        
        for indicator in simple_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                return 0.0  # Very simple query
        
        complexity_score = sum(bool(re.search(indicator, query, re.IGNORECASE)) for indicator in complexity_indicators)
        return min(complexity_score / len(complexity_indicators), 1.0)

    def _calculate_context_length(self, conversation_history: list) -> int:
        """
        Calculate the total length of the conversation history.
        """
        return sum(len(message['content']) for message in conversation_history)

    def _identify_task_type(self, query: str) -> str:
        """
        Identify the type of task based on the query.
        """
        if re.search(r'\b(code|program|function|algorithm)\b', query, re.IGNORECASE):
            return 'coding'
        elif re.search(r'\b(analyze|evaluate|compare|contrast)\b', query, re.IGNORECASE):
            return 'analysis'
        elif re.search(r'\b(creative|write|compose|design)\b', query, re.IGNORECASE):
            return 'creative'
        elif re.search(r'^\s*\d+\s*[\+\-\*/]\s*\d+\s*$', query):
            return 'arithmetic'
        else:
            return 'general'

    def _get_local_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for local model processing.
        """
        return {
            'model': self.local_model,
            'max_tokens': 100,
            'temperature': 0.2,
            'top_p': 0.9,
            'system_message': self._get_system_message(task_type, 'local'),
            'use_local_model': True
        }

    def _get_low_tier_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for low-tier model processing.
        """
        return {
            'model': self.models['4']['id'],  # Claude 3 Haiku
            'max_tokens': 250,
            'temperature': 0.5,
            'top_p': 0.95,
            'system_message': self._get_system_message(task_type, 'low')
        }

    def _get_mid_tier_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for mid-tier model processing.
        """
        return {
            'model': self.models['3']['id'],  # Claude 3 Sonnet
            'max_tokens': 1000,
            'temperature': 0.7,
            'top_p': 0.95,
            'system_message': self._get_system_message(task_type, 'mid')
        }

    def _get_high_tier_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get configuration for high-tier model processing.
        """
        return {
            'model': self.models['1']['id'],  # Claude 3.5 Sonnet
            'max_tokens': 2000,
            'temperature': 0.9,
            'top_p': 1.0,
            'system_message': self._get_system_message(task_type, 'high')
        }

    def _get_system_message(self, task_type: str, tier: str) -> str:
        """
        Get an appropriate system message based on the task type and model tier.
        """
        base_message = "You are an AI assistant. "
        
        if task_type == 'coding':
            base_message += "Provide clear, efficient, and well-commented code solutions. "
        elif task_type == 'analysis':
            base_message += "Offer detailed analysis and evaluation of the given topic. "
        elif task_type == 'creative':
            base_message += "Be creative and innovative in your responses. "
        elif task_type == 'arithmetic':
            base_message += "Provide quick and accurate arithmetic calculations. "
        
        if tier == 'local':
            base_message += "Focus on providing concise and direct answers. "
        elif tier == 'high':
            base_message += "Utilize your full capabilities to provide comprehensive and nuanced responses. "
        
        return base_message

advanced_router = AdvancedRouter()
