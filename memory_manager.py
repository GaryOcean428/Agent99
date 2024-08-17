import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import redis
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class MemoryManager:
    def __init__(self):
        """Initialize the MemoryManager with necessary configurations and connections."""
        self.redis_client = self._setup_redis()
        self.mongo_client, self.mongo_db, self.mongo_collection = self._setup_mongodb()
        self.pinecone_index = self._setup_pinecone()
        self.short_term_memory: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer()
        self.configure_threshold = float(os.getenv("CONFIGURE_THRESHOLD", "0.5"))

    def _setup_redis(self) -> Optional[redis.Redis]:
        """Set up and return a Redis client connection."""
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD")
        try:
            redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                ssl=True,
                decode_responses=True,
            )
            redis_client.ping()
            logger.info("Successfully connected to Redis")
            return redis_client
        except redis.RedisError as e:
            logger.error(
                f"Failed to connect to Redis: {str(e)}. Caching will be disabled."
            )
            return None

    def _setup_mongodb(self):
        """Set up and return a MongoDB client connection."""
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            try:
                mongo_client = MongoClient(mongo_uri, server_api=ServerApi("1"))
                mongo_client.admin.command("ping")
                mongo_db = mongo_client[os.getenv("MONGODB_DB_NAME", "chat99")]
                mongo_collection = mongo_db["long_term_memory"]
                logger.info("Successfully connected to MongoDB")
                return mongo_client, mongo_db, mongo_collection
            except Exception as e:
                logger.error(f"An error occurred while setting up MongoDB: {str(e)}")
        return None, None, None

    def _setup_pinecone(self):
        """Set up and return a Pinecone index."""
        try:
            pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = os.getenv("PINECONE_INDEX_NAME", "chat99-index")
            if index_name not in pinecone.list_indexes().names():
                pinecone.create_index(
                    name=index_name,
                    dimension=768,  # Adjust based on your embedding size
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-west-2"),
                )
            index = pinecone.Index(index_name)
            logger.info(f"Successfully connected to Pinecone index: {index_name}")
            return index
        except Exception as e:
            logger.error(f"An error occurred while setting up Pinecone: {str(e)}")
            return None

    def add_to_short_term(self, message: Dict[str, str]):
        """Add a message to short-term memory."""
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > 10:  # Adjust as needed
            self.short_term_memory.pop(0)

    def update_memory(self, user_input: str, response: str):
        """Update the memory with the latest interaction."""
        self.add_to_short_term({"role": "user", "content": user_input})
        self.add_to_short_term({"role": "assistant", "content": response})

        topic = self._extract_topic(user_input + " " + response)
        summary = self._generate_summary(user_input, response)
        self._update_long_term_memory(topic, summary)

        if self.redis_client:
            try:
                self.redis_client.setex(f"{user_input}_last_response", 3600, response)
            except redis.RedisError as e:
                logger.error(f"Redis error: {str(e)}")

    def get_relevant_context(self, user_input: str) -> str:
        """Retrieve relevant context from memory based on the query."""
        relevant_info = []
        key_words = self._extract_key_words(user_input)

        for message in reversed(self.short_term_memory):
            if any(word in message["content"].lower() for word in key_words):
                relevant_info.append(message["content"])

        long_term_context = self._get_long_term_memories(user_input, limit=5)
        relevant_info.extend(long_term_context)

        pinecone_context = self._search_pinecone(user_input)
        if pinecone_context:
            relevant_info.append(pinecone_context)

        if any(
            phrase in user_input.lower()
            for phrase in [
                "what else",
                "what have we discussed",
                "what did we talk about",
            ]
        ):
            recap = self._generate_conversation_recap()
            relevant_info.append(f"Conversation Recap: {recap}")

        return ". ".join(relevant_info)

    def get_cached_response(self, user_input: str) -> str:
        """Retrieve cached response from Redis."""
        if self.redis_client:
            try:
                return self.redis_client.get(f"{user_input}_last_response") or ""
            except redis.RedisError as e:
                logger.error(f"Redis error: {str(e)}")
        return ""

    def _extract_key_words(self, text: str) -> List[str]:
        """Extract key words from a text string."""
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text.lower())
        return [
            word for word in word_tokens if word not in stop_words and word.isalnum()
        ]

    def _update_long_term_memory(self, topic: str, summary: str):
        """Update the long-term memory with a new entry."""
        if self.mongo_collection:
            try:
                relevance_score = self._calculate_relevance(topic, summary)
                if relevance_score >= self.configure_threshold:
                    self.mongo_collection.insert_one(
                        {
                            "topic": topic,
                            "summary": summary,
                            "timestamp": datetime.now(),
                            "relevance": relevance_score,
                        }
                    )
                    logger.info(f"Added new memory: {topic}")
                else:
                    logger.info(f"Memory not added due to low relevance: {topic}")
            except Exception as e:
                logger.error(f"Error updating long-term memory: {str(e)}")

    def _calculate_relevance(self, topic: str, summary: str) -> float:
        """Calculate the relevance of a memory entry."""
        combined_text = f"{topic} {summary}"
        word_count = len(combined_text.split())
        return min(1.0, word_count / 100)  # Simple scaling based on word count

    def _get_long_term_memories(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve long-term memories from MongoDB based on a query."""
        if self.mongo_collection:
            try:
                memories = list(
                    self.mongo_collection.find(
                        {"$text": {"$search": query}}, {"score": {"$meta": "textScore"}}
                    )
                    .sort([("score", {"$meta": "textScore"})])
                    .limit(limit)
                )
                return [
                    f"Topic: {m['topic']}, Content: {m['summary']}" for m in memories
                ]
            except Exception as e:
                logger.error(f"Error retrieving long-term memories: {str(e)}")
        return []

    def _search_pinecone(self, query: str) -> Optional[str]:
        """Search for relevant context in Pinecone."""
        if not self.pinecone_index:
            return None
        try:
            vector = self.vectorizer.transform([query]).toarray()[0].tolist()
            results = self.pinecone_index.query(
                vector=vector, top_k=1, include_metadata=True
            )
            if results and "matches" in results and results["matches"]:
                return results["matches"][0].get("metadata", {}).get("response")
            else:
                logger.warning("No matching results found in Pinecone")
                return None
        except Exception as e:
            logger.error(f"An error occurred while searching Pinecone: {str(e)}")
            return None

    def _extract_topic(self, text: str) -> str:
        """Extract a topic from the provided text."""
        words = self._extract_key_words(text)
        return " ".join(words[:3])  # Use top 3 keywords as topic

    def _generate_summary(self, user_input: str, response: str) -> str:
        """Generate a summary for the provided interaction."""
        combined = user_input + " " + response
        return combined[:200] + "..." if len(combined) > 200 else combined

    def _generate_conversation_recap(self) -> str:
        """Generate a recap of the conversation."""
        topics = set()
        for memory in self.short_term_memory + self._get_long_term_memories(
            "", limit=10
        ):
            topics.add(memory.get("topic", ""))
        return f"We've discussed these topics: {', '.join(topics)}"

    def cleanup_memory(self):
        """Clean up expired memory from MongoDB."""
        if self.mongo_collection:
            try:
                cutoff_date = datetime.now() - timedelta(days=30)  # Adjust as needed
                self.mongo_collection.delete_many({"timestamp": {"$lt": cutoff_date}})
                logger.info("Cleaned up old memories")
            except Exception as e:
                logger.error(f"Error cleaning up memory: {str(e)}")


# Instantiate the MemoryManager
memory_manager = MemoryManager()
