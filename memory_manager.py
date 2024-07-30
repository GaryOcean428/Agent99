"""
MemoryManager: Handles storage and retrieval of conversation context and user information.
"""

from typing import Dict, List
import re
from datetime import datetime, timedelta
from pymongo import MongoClient
import redis
from config import (
    MONGODB_URI, MONGODB_DB_NAME, REDIS_HOST, REDIS_PORT, REDIS_DB,
    MAX_SHORT_TERM_MEMORY, LONG_TERM_MEMORY_EXPIRY_DAYS
)

class MemoryManager:
    def __init__(self):
        self.mongo_client = MongoClient(MONGODB_URI)
        self.db = self.mongo_client[MONGODB_DB_NAME]
        self.long_term_memory = self.db.long_term_memory

        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        self.short_term_memory = []

    def add_to_short_term(self, message: Dict[str, str]):
        """Add a message to short-term memory."""
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > MAX_SHORT_TERM_MEMORY:
            self.short_term_memory.pop(0)

    def update_memory(self, user_input: str, response: str):
        """Update both short-term and long-term memory."""
        self.add_to_short_term({"role": "user", "content": user_input})
        self.add_to_short_term({"role": "assistant", "content": response})

        key_words = self._extract_key_words(user_input)
        for word in key_words:
            self.long_term_memory.update_one(
                {"keyword": word},
                {"$set": {
                    "context": response[:100],
                    "timestamp": datetime.now(),
                    "relevance": 1.0
                }},
                upsert=True
            )

    def get_relevant_context(self, user_input: str) -> str:
        """Retrieve relevant context from both short-term and long-term memory."""
        relevant_info = []
        key_words = self._extract_key_words(user_input)

        # Check short-term memory
        for message in reversed(self.short_term_memory):
            if any(word in message["content"].lower() for word in key_words):
                relevant_info.append(message["content"])

        # Check long-term memory
        for word in key_words:
            memory = self.long_term_memory.find_one({"keyword": word})
            if memory:
                relevant_info.append(f"{word}: {memory['context']}")
                self._update_relevance(word)

        return ". ".join(relevant_info)

    def determine_complexity(self, user_input: str) -> str:
        """Determine the complexity of the query."""
        word_count = len(user_input.split())
        if word_count > 50 or any(word in user_input.lower() for word in ["complex", "difficult", "advanced"]):
            return "high"
        elif word_count > 20:
            return "mid"
        else:
            return "low"

    def _extract_key_words(self, text: str) -> List[str]:
        """Extract key words from the given text."""
        words = re.findall(r'\b\w{4,}\b', text.lower())
        return list(set(words))  # Remove duplicates

    def _update_relevance(self, word: str):
        """Update the relevance score of a word in long-term memory."""
        memory = self.long_term_memory.find_one({"keyword": word})
        if memory:
            time_diff = datetime.now() - memory['timestamp']
            new_relevance = max(0.1, 1.0 - (time_diff.total_seconds() / (LONG_TERM_MEMORY_EXPIRY_DAYS * 24 * 3600)))
            self.long_term_memory.update_one(
                {"keyword": word},
                {"$set": {"relevance": new_relevance}}
            )

    def cleanup_memory(self):
        """Remove old and irrelevant entries from long-term memory."""
        expiry_date = datetime.now() - timedelta(days=LONG_TERM_MEMORY_EXPIRY_DAYS)
        self.long_term_memory.delete_many({
            "$or": [
                {"timestamp": {"$lt": expiry_date}},
                {"relevance": {"$lt": 0.1}}
            ]
        })

memory_manager = MemoryManager()