import os
import requests
from typing import Dict, List
import re
from datetime import datetime, timedelta
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
        self.short_term_memory = []

        # TF-IDF Vectorizer for similarity search
        self.vectorizer = TfidfVectorizer()

    def add_to_short_term(self, message: Dict[str, str]):
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > 10:  # Adjust as needed
            self.short_term_memory.pop(0)

    def update_memory(self, user_input: str, response: str):
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
                relevant_info.append(
                    f"Topic: {memory.get('topic', 'Unknown')}, Content: {memory.get('summary', 'No summary available')}"
                )
        except Exception as e:
            print(f"Error retrieving long-term memories: {str(e)}")

        # If asked about previous discussions, provide a recap
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
        if self.redis_client:
            try:
                return self.redis_client.get(f"{user_input}_last_response") or ""
            except redis.RedisError as e:
                print(f"Redis error: {str(e)}")
        return ""

    def determine_complexity(self, user_input: str) -> str:
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
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text.lower())
        return [
            word for word in word_tokens if word not in stop_words and word.isalnum()
        ]

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
            print(f"Error retrieving long-term memories: {str(e)}")
            return []

    def _extract_topic(self, text: str) -> str:
        # Simple topic extraction, can be improved with more sophisticated NLP
        words = self._extract_key_words(text)
        return " ".join(words[:3])  # Use top 3 keywords as topic

    def _generate_summary(self, user_input: str, response: str) -> str:
        # Simple summary generation, can be improved
        combined = user_input + " " + response
        return combined[:200] + "..." if len(combined) > 200 else combined

    def _generate_conversation_recap(self):
        topics = set()
        for memory in self.short_term_memory + self._get_long_term_memories(
            "", limit=10
        ):
            topics.add(memory.get("topic", ""))
        return f"We've discussed these topics: {', '.join(topics)}"

    def cleanup_memory(self):
        if self.mongo_collection:
            try:
                cutoff_date = datetime.now() - timedelta(days=30)  # Adjust as needed
                self.mongo_collection.delete_many({"timestamp": {"$lt": cutoff_date}})
            except Exception as e:
                print(f"Error cleaning up memory: {str(e)}")


memory_manager = MemoryManager()
