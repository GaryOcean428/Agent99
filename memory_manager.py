"""
MemoryManager: Handles storage and retrieval of conversation context and user information.
"""

from typing import Dict, List
import re
from datetime import datetime, timedelta

class MemoryManager:
    """Manages the memory for the chat assistant."""

    def __init__(self):
        """Initialize the MemoryManager with an empty memory dictionary."""
        self.memory: Dict[str, Dict[str, any]] = {}
        self.short_term_memory: List[Dict[str, str]] = []

    def add_to_short_term(self, message: Dict[str, str]):
        """
        Add a message to the short-term memory.

        Args:
            message (Dict[str, str]): The message to add to short-term memory.
        """
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)

    def update_memory(self, user_input: str, response: str):
        """
        Update the memory with key information from the conversation.

        Args:
            user_input (str): The user's input message.
            response (str): The assistant's response message.
        """
        key_words = self._extract_key_words(user_input)
        for word in key_words:
            self.memory[word] = {
                'context': response[:100],  # Store first 100 characters of response
                'timestamp': datetime.now(),
                'relevance': 1.0
            }

    def get_relevant_context(self, user_input: str) -> str:
        """
        Retrieve relevant context based on the user's input.

        Args:
            user_input (str): The user's input message.

        Returns:
            str: Relevant context from memory.
        """
        relevant_info = []
        key_words = self._extract_key_words(user_input)
        for word in key_words:
            if word in self.memory:
                info = self.memory[word]
                relevant_info.append(f"{word}: {info['context']}")
                self._update_relevance(word)
        return ". ".join(relevant_info)

    def is_relevant(self, content: str) -> bool:
        """
        Check if the given content is relevant based on the current memory.

        Args:
            content (str): The content to check for relevance.

        Returns:
            bool: True if the content is relevant, False otherwise.
        """
        key_words = self._extract_key_words(content)
        return any(word in self.memory and self.memory[word]['relevance'] > 0.5 for word in key_words)

    def _extract_key_words(self, text: str) -> List[str]:
        """Extract key words from the given text."""
        # This is a simple implementation. Consider using NLP techniques for better extraction.
        words = re.findall(r'\b\w{4,}\b', text.lower())
        return list(set(words))  # Remove duplicates

    def _update_relevance(self, word: str):
        """Update the relevance score of a word in memory."""
        if word in self.memory:
            time_diff = datetime.now() - self.memory[word]['timestamp']
            # Decrease relevance over time, but not below 0.1
            self.memory[word]['relevance'] = max(0.1, 1.0 - (time_diff.total_seconds() / (24 * 3600)))

    def cleanup_memory(self):
        """Remove old and irrelevant entries from memory."""
        current_time = datetime.now()
        self.memory = {
            word: info for word, info in self.memory.items()
            if current_time - info['timestamp'] <= timedelta(days=7) or info['relevance'] > 0.3
        }

memory_manager = MemoryManager()
