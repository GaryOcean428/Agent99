<<<<<<< HEAD
import os
import requests
=======
"""
This module provides a MemoryManager class for managing short-term and long-term memory
in a chatbot system, including caching and retrieval of relevant information.
"""

import os
import logging
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
from typing import Dict, List
from datetime import datetime, timedelta
<<<<<<< HEAD
import redis
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class MemoryManager:
=======

try:
    import redis
except ImportError:
    redis = None
    print("Redis not installed. Please install it using 'pip install redis'")

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None
    print("PyMongo not installed. Please install it using 'pip install pymongo'")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except ImportError:
    nltk = None
    print("NLTK not installed. Please install it using 'pip install nltk'")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    print(
        "Scikit-learn not installed. Please install it using 'pip install scikit-learn'"
    )

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages short-term and long-term memory for a chatbot, including caching and
    retrieval of relevant information.
    """

>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
    def __init__(self):
        self.api_key = os.getenv("MONGO_DATA_API_KEY")
        self.api_url = "https://ap-southeast-2.aws.data.mongodb-api.com/app/data-oqozfgx/endpoint/data/v1/action/"
        self.headers = {
            "Content-Type": "application/json",
            "Access-Control-Request-Headers": "*",
            "api-key": self.api_key,
        }
        self.database = "chat99"
        self.collection = "long_term_memory"
        self.data_source = "GaryCluster0"

<<<<<<< HEAD
        # Redis connection
        redis_host = os.getenv("REDIS_HOST", "real-wren-52199.upstash.io")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD")
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                ssl=True,
                decode_responses=True,
            )
            self.redis_client.ping()  # Test the connection
        except redis.ConnectionError:
            print("Failed to connect to Redis. Caching will be disabled.")
            self.redis_client = None

        # MongoDB connection
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri and "directConnection=true" in mongo_uri:
            mongo_uri = mongo_uri.replace("directConnection=true", "")
            if mongo_uri.endswith("&"):
                mongo_uri = mongo_uri[:-1]

        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection = None

        if mongo_uri:
            try:
                self.mongo_client = MongoClient(mongo_uri)
                self.mongo_db = self.mongo_client[self.database]
                self.mongo_collection = self.mongo_db[self.collection]
            except Exception as e:
                print(f"Error connecting to MongoDB: {str(e)}")

        # Short-term memory
=======
        self.redis_client = self._setup_redis()
        self.mongo_client, self.mongo_db, self.mongo_collection = self._setup_mongodb()
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        self.short_term_memory = []
        self.vectorizer = TfidfVectorizer() if TfidfVectorizer else None

<<<<<<< HEAD
        # TF-IDF Vectorizer for similarity search
        self.vectorizer = TfidfVectorizer()

    def add_to_short_term(self, message: Dict[str, str]):
=======
    def _setup_redis(self):
        """Set up Redis connection."""
        if not redis:
            return None

        redis_host = os.getenv("REDIS_HOST", "real-wren-52199.upstash.io")
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
            return redis_client
        except redis.RedisError as e:
            logger.error(
                "Failed to connect to Redis: %s. Caching will be disabled.", str(e)
            )
            return None

    def _setup_mongodb(self):
        """Set up MongoDB connection."""
        if not MongoClient:
            return None, None, None

        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri and "directConnection=true" in mongo_uri:
            mongo_uri = mongo_uri.replace("directConnection=true", "")
            if mongo_uri.endswith("&"):
                mongo_uri = mongo_uri[:-1]

        if mongo_uri:
            try:
                mongo_client = MongoClient(mongo_uri)
                mongo_db = mongo_client[self.database]
                mongo_collection = mongo_db[self.collection]
                return mongo_client, mongo_db, mongo_collection
            except Exception as e:
                logger.error("Error connecting to MongoDB: %s", str(e))
        return None, None, None

    def add_to_short_term(self, message: Dict[str, str]):
        """Add a message to short-term memory, removing oldest if limit is reached."""
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > 10:  # Adjust as needed
            self.short_term_memory.pop(0)

    def update_memory(self, user_input: str, response: str):
<<<<<<< HEAD
=======
        """Update both short-term and long-term memory with new interaction."""
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        self.add_to_short_term({"role": "user", "content": user_input})
        self.add_to_short_term({"role": "assistant", "content": response})

        topic = self._extract_topic(user_input + " " + response)
        summary = self._generate_summary(user_input, response)
        self._update_long_term_memory(topic, summary)

        if self.redis_client:
            try:
                self.redis_client.setex(f"{user_input}_last_response", 3600, response)
            except redis.RedisError as e:
<<<<<<< HEAD
                print(f"Redis error: {str(e)}")

    def get_relevant_context(self, user_input: str) -> str:
=======
                logger.error("Redis error: %s", str(e))

    def get_relevant_context(self, user_input: str) -> str:
        """Retrieve relevant context based on user input from both short-term and long-term memory."""
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        relevant_info = []
        key_words = self._extract_key_words(user_input)

        for message in reversed(self.short_term_memory):
            if any(word in message["content"].lower() for word in key_words):
                relevant_info.append(message["content"])

<<<<<<< HEAD
        # Check long-term memory
        try:
            long_term_memories = self._get_long_term_memories(user_input, limit=5)
            for memory in long_term_memories:
                relevant_info.append(
                    f"Topic: {memory.get('topic', 'Unknown')}, Content: {memory.get('summary', 'No summary available')}"
                )
        except Exception as e:
            print(f"Error retrieving long-term memories: {str(e)}")

        # If asked about previous discussions, provide a recap
=======
        try:
            long_term_memories = self._get_long_term_memories(user_input, limit=5)
            for memory in long_term_memories:
                topic = memory.get("topic", "Unknown")
                summary = memory.get("summary", "No summary available")
                relevant_info.append(f"Topic: {topic}, Content: {summary}")
        except Exception as e:
            logger.error("Error retrieving long-term memories: %s", str(e))

>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
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
<<<<<<< HEAD
=======
        """Retrieve cached response for the given user input."""
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        if self.redis_client:
            try:
                return self.redis_client.get(f"{user_input}_last_response") or ""
            except redis.RedisError as e:
<<<<<<< HEAD
                print(f"Redis error: {str(e)}")
        return ""

    def determine_complexity(self, user_input: str) -> str:
=======
                logger.error("Redis error: %s", str(e))
        return ""

    def determine_complexity(self, user_input: str) -> str:
        """Determine the complexity of the user input."""
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        word_count = len(user_input.split())
        if word_count > 50 or any(
            word in user_input.lower() for word in ["complex", "difficult", "advanced"]
        ):
            return "high"
        elif word_count > 20:
            return "mid"
        else:
            return "low"

    def _extract_key_words(self, text: str) -> List[str]:
<<<<<<< HEAD
=======
        """Extract key words from the given text, removing stop words."""
        if not nltk:
            return text.lower().split()

>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text.lower())
        return [
            word for word in word_tokens if word not in stop_words and word.isalnum()
        ]

<<<<<<< HEAD
    def _update_long_term_memory(self, topic: str, summary: str):
        if self.mongo_collection:
            try:
                self.mongo_collection.insert_one(
                    {
                        "topic": topic,
                        "summary": summary,
                        "timestamp": datetime.now(),
                        "relevance": 1.0,
                    }
                )
            except Exception as e:
                print(f"Error updating long-term memory: {str(e)}")

    def _get_long_term_memories(self, query: str, limit: int = 5) -> List[Dict]:
        if not self.mongo_collection:
=======
    def _get_long_term_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant long-term memories based on the query."""
        if self.mongo_collection is None or self.vectorizer is None:
            logger.error("MongoDB collection or vectorizer is not initialized")
            return []

        try:
            all_memories = list(self.mongo_collection.find().sort("timestamp", -1).limit(100))

            if not all_memories:
                return []

            memory_texts = [f"{m.get('topic', '')} {m.get('summary', '')}" for m in all_memories]
            query_vector = self.vectorizer.fit_transform([query])
            memory_vectors = self.vectorizer.transform(memory_texts)

            similarities = cosine_similarity(query_vector, memory_vectors).flatten()
            top_indices = similarities.argsort()[-limit:][::-1]

            return [all_memories[i] for i in top_indices]
        except Exception as e:
            logger.error("Error retrieving long-term memories: %s", str(e))
            return []

    try:
        self.mongo_collection.insert_one(
            {
                "topic": topic,
                "summary": summary,
                "timestamp": datetime.now(),
                "relevance": 1.0,
            }
        )
    except Exception as e:
        logger.error("Error updating long-term memory: %s", str(e))

    def _get_long_term_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant long-term memories based on the query."""
        if self.mongo_collection is None or self.vectorizer is None:
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
            return []

        try:
            all_memories = list(
                self.mongo_collection.find().sort("timestamp", -1).limit(100)
            )

            if not all_memories:
                return []

            memory_texts = [
                f"{m.get('topic', '')} {m.get('summary', '')}" for m in all_memories
            ]
            query_vector = self.vectorizer.fit_transform([query])
            memory_vectors = self.vectorizer.transform(memory_texts)

            similarities = cosine_similarity(query_vector, memory_vectors).flatten()
            top_indices = similarities.argsort()[-limit:][::-1]

            return [all_memories[i] for i in top_indices]
        except Exception as e:
<<<<<<< HEAD
            print(f"Error retrieving long-term memories: {str(e)}")
            return []

    def _extract_topic(self, text: str) -> str:
        # Simple topic extraction, can be improved with more sophisticated NLP
=======
            logger.error(f"Error retrieving long-term memories: {str(e)}")
            return []

    def _extract_topic(self, text: str) -> str:
        """Extract a topic from the given text."""
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        words = self._extract_key_words(text)
        return " ".join(words[:3])  # Use top 3 keywords as topic

    def _generate_summary(self, user_input: str, response: str) -> str:
<<<<<<< HEAD
        # Simple summary generation, can be improved
        combined = user_input + " " + response
        return combined[:200] + "..." if len(combined) > 200 else combined

    def _generate_conversation_recap(self):
=======
        """Generate a summary of the user input and response."""
        combined = user_input + " " + response
        return combined[:200] + "..." if len(combined) > 200 else combined

    def _generate_conversation_recap(self) -> str:
        """Generate a recap of the conversation topics."""
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        topics = set()
        for memory in self.short_term_memory + self._get_long_term_memories(
            "", limit=10
        ):
            topics.add(memory.get("topic", ""))
        return f"We've discussed these topics: {', '.join(topics)}"

    def cleanup_memory(self):
<<<<<<< HEAD
=======
        """Clean up old memories from long-term storage."""
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b
        if self.mongo_collection:
            try:
                cutoff_date = datetime.now() - timedelta(days=30)  # Adjust as needed
                self.mongo_collection.delete_many({"timestamp": {"$lt": cutoff_date}})
            except Exception as e:
<<<<<<< HEAD
                print(f"Error cleaning up memory: {str(e)}")
=======
                logger.error("Error cleaning up memory: %s", str(e))
>>>>>>> fa07d026fe41eb63d9374cd5987c9eebbb28c44b


memory_manager = MemoryManager()
def _update_long_term_memory(self, topic: str, summary: str):
        """Update long-term memory with new topic and summary."""
        if self.mongo_collection:
            try:
                self.mongo_collection.insert_one({
                    "topic": topic,
                    "summary": summary,
                    "timestamp": datetime.now(),
                    "relevance": 1.0,
                })
            except Exception as e:
                logger.error("Error updating long-term memory: %s", str(e))

    def _get_long_term_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant long-term memories based on the query."""
        if self.mongo_collection is None or self.vectorizer is None:
            logger.error("MongoDB collection or vectorizer is not initialized")
            return []

        try:
            all_memories = list(self.mongo_collection.find().sort("timestamp", -1).limit(100))

            if not all_memories:
                return []

            memory_texts = [f"{m.get('topic', '')} {m.get('summary', '')}" for m in all_memories]
            query_vector = self.vectorizer.fit_transform([query])
            memory_vectors = self.vectorizer.transform(memory_texts)

            similarities = cosine_similarity(query_vector, memory_vectors).flatten()
            top_indices = similarities.argsort()[-limit:][::-1]

            return [all_memories[i] for i in top_indices]
        except Exception as e:
            logger.error("Error retrieving long-term memories: %s", str(e))
            return []
        """Retrieve relevant long-term memories based on the query."""
        if self.mongo_collection is None or self.vectorizer is None:
            logger.error("MongoDB collection or vectorizer is not initialized")
            return []

        try:
            all_memories = list(self.mongo_collection.find().sort("timestamp", -1).limit(100))

            if not all_memories:
                return []

            memory_texts = [f"{m.get('topic', '')} {m.get('summary', '')}" for m in all_memories]
            query_vector = self.vectorizer.fit_transform([query])
            memory_vectors = self.vectorizer.transform(memory_texts)

            similarities = cosine_similarity(query_vector, memory_vectors).flatten()
            top_indices = similarities.argsort()[-limit:][::-1]

            return [all_memories[i] for i in top_indices]
        except Exception as e:
            logger.error("Error retrieving long-term memories: %s", str(e))
            return []