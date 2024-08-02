"""
MemoryManager: A module for managing short-term and long-term memory in a chatbot system.

This module provides functionality for storing, retrieving, and managing conversational
context using both in-memory storage and external databases (MongoDB and Redis).
"""

import os
from typing import Dict, List
from datetime import datetime, timedelta

try:
    import redis
except ImportError:
    print("Redis not installed. Please install it using 'pip install redis'")
    redis = None

try:
    from pymongo import MongoClient
except ImportError:
    print("PyMongo not installed. Please install it using 'pip install pymongo'")
    MongoClient = None

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except ImportError:
    print("NLTK not installed. Please install it using 'pip install nltk'")
    nltk = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Scikit-learn not installed. Please install it using 'pip install scikit-learn'")
    TfidfVectorizer = None
    cosine_similarity = None

class MemoryManager:
    """
    Manages short-term and long-term memory for a chatbot, including caching and
    retrieval of relevant information.
    """

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

        # Redis connection
        self.redis_client = self._setup_redis()

        # MongoDB connection
        self.mongo_client, self.mongo_db, self.mongo_collection = self._setup_mongodb()

        # Short-term memory
        self.short_term_memory = []

        # TF-IDF Vectorizer for similarity search
        self.vectorizer = TfidfVectorizer() if TfidfVectorizer else None

    def _setup_redis(self):
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
            redis_client.ping()  # Test the connection
            return redis_client
        except (redis.ConnectionError, redis.RedisError) as e:
            print(f"Failed to connect to Redis: {str(e)}. Caching will be disabled.")
            return None

    def _setup_mongodb(self):
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
                print(f"Error connecting to MongoDB: {str(e)}")
        return None, None, None

    def add_to_short_term(self, message: Dict[str, str]):
        """Add a message to short-term memory, removing oldest if limit is reached."""
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > 10:  # Adjust as needed
            self.short_term_memory.pop(0)

    def search_memories(self, query: str) -> List[Dict]:
        if not self.mongo_collection or not self.vectorizer:
            return []
    
        try:
            all_memories = list(self.mongo_collection.find().sort("timestamp", -1).limit(100))
    
            if not all_memories:
                return []
    
            memory_texts = [f"{m.get('topic', '')} {m.get('summary', '')}" for m in all_memories]
            query_vector = self.vectorizer.transform([query])
            memory_vectors = self.vectorizer.transform(memory_texts)
    
            similarities = cosine_similarity(query_vector, memory_vectors).flatten()
            top_indices = similarities.argsort()[-5:][::-1]  # Get top 5 most relevant memories
    
            return [all_memories[i] for i in top_indices]
        import logging
        
        except Exception as e:
            logging.error(f"Error searching memories: {str(e)}")
            return []

    def update_memory(self, user_input: str, response: str):
        """Update both short-term and long-term memory with new interaction."""
        self.add_to_short_term({"role": "user", "content": user_input})
        self.add_to_short_term({"role": "assistant", "content": response})

        topic = self._extract_topic(user_input + " " + response)
        summary = self._generate_summary(user_input, response)
        self._update_long_term_memory(topic, summary)

        if self.redis_client:
            try:
                self.redis_client.setex(f"{user_input}_last_response", 3600, response)
            except redis.RedisError as e:
                print(f"Redis error: {str(e)}")

    def get_relevant_context(self, user_input: str) -> str:
        """Retrieve relevant context based on user input from both short-term and long-term memory."""
        relevant_info = []
        key_words = self._extract_key_words(user_input)

        # Check short-term memory
        for message in reversed(self.short_term_memory):
            if any(word in message["content"].lower() for word in key_words):
                relevant_info.append(message["content"])

        # Check long-term memory
        try:
            long_term_memories = self._get_long_term_memories(user_input, limit=5)
            for memory in long_term_memories:
                topic = memory.get('topic', 'Unknown')
                summary = memory.get('summary', 'No summary available')
                relevant_info.append(f"Topic: {topic}, Content: {summary}")
        except Exception as e:
            print(f"Error retrieving long-term memories: {str(e)}")

        # If asked about previous discussions, provide a recap
        if any(phrase in user_input.lower() for phrase in [
            "what else", "what have we discussed", "what did we talk about"
        ]):
            recap = self._generate_conversation_recap()
            relevant_info.append(f"Conversation Recap: {recap}")

        return ". ".join(relevant_info)

    def get_cached_response(self, user_input: str) -> str:
        """Retrieve cached response for the given user input."""
        if self.redis_client:
            try:
                return self.redis_client.get(f"{user_input}_last_response") or ""
            except redis.RedisError as e:
                print(f"Redis error: {str(e)}")
        return ""

    def determine_complexity(self, user_input: str) -> str:
        """Determine the complexity of the user input."""
        word_count = len(user_input.split())
        if word_count > 50 or any(word in user_input.lower() for word in ["complex", "difficult", "advanced"]):
            return "high"
        elif word_count > 20:
            return "mid"
        else:
            return "low"

    def _extract_key_words(self, text: str) -> List[str]:
        """Extract key words from the given text, removing stop words."""
        if not nltk:
            return text.lower().split()
        
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text.lower())
        return [word for word in word_tokens if word not in stop_words and word.isalnum()]

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
                print(f"Error updating long-term memory: {str(e)}")

    def _get_long_term_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant long-term memories based on the query."""
        if not self.mongo_collection or not self.vectorizer:
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
            print(f"Error retrieving long-term memories: {str(e)}")
            return []

    def _extract_topic(self, text: str) -> str:
        """Extract a topic from the given text."""
        words = self._extract_key_words(text)
        return " ".join(words[:3])  # Use top 3 keywords as topic

    def _generate_summary(self, user_input: str, response: str) -> str:
        """Generate a summary of the user input and response."""
        combined = user_input + " " + response
        return combined[:200] + "..." if len(combined) > 200 else combined

    def _generate_conversation_recap(self) -> str:
        """Generate a recap of the conversation topics."""
        topics = set()
        for memory in self.short_term_memory + self._get_long_term_memories("", limit=10):
            topics.add(memory.get('topic', ''))
        return f"We've discussed these topics: {', '.join(topics)}"

    def cleanup_memory(self):
        """Clean up old memories from long-term storage."""
        if self.mongo_collection:
            try:
                cutoff_date = datetime.now() - timedelta(days=30)  # Adjust as needed
                self.mongo_collection.delete_many({"timestamp": {"$lt": cutoff_date}})
            except Exception as e:
                print(f"Error cleaning up memory: {str(e)}")

memory_manager = MemoryManager()