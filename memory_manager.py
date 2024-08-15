"""
This module provides a MemoryManager class for managing short-term and long-term memory
in a chatbot system, including caching and retrieval of relevant information.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymongo import MongoClient, errors as pymongo_errors
from pymongo.server_api import ServerApi
import redis
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load configuration from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "128"))
MONGO_URI = os.getenv("MONGO_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
CONFIGURE_THRESHOLD = float(os.getenv("CONFIGURE_THRESHOLD", "0.5"))


class MemoryManager:
    def __init__(self):
        """Initialize the MemoryManager with necessary configurations and connections."""
        self.redis_client = self._setup_redis()
        self.mongo_client, self.mongo_db, self.mongo_collection = self._setup_mongodb()
        self.pinecone_index = self._setup_pinecone()
        self.short_term_memory: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(max_features=PINECONE_DIMENSION)
        self.configure_threshold = CONFIGURE_THRESHOLD
        self._fit_vectorizer()

    def _setup_redis(self) -> Optional[redis.Redis]:
        """Set up and return a Redis client connection."""
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                ssl=True,
                decode_responses=True,
            )
            redis_client.ping()
            logger.info("Successfully connected to Redis")
            return redis_client
        except redis.RedisError as e:
            logger.error(
                "Failed to connect to Redis: %s. Caching will be disabled.", str(e)
            )
            return None

    def _setup_mongodb(self):
        """Set up and return a MongoDB client connection."""
        try:
            mongo_client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
            mongo_client.admin.command("ping")
            mongo_db = mongo_client[MONGODB_DB_NAME]
            mongo_collection = mongo_db["long_term_memory"]
            # Ensure a text index exists
            mongo_collection.create_index(
                [("user_input", "text"), ("response", "text")]
            )
            logger.info("Successfully connected to MongoDB")
            return mongo_client, mongo_db, mongo_collection
        except pymongo_errors.PyMongoError as e:
            logger.error("An error occurred while setting up MongoDB: %s", str(e))
            return None, None, None

    def _setup_pinecone(self):
        """Set up and return a Pinecone index."""
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            # Check if the index exists
            if PINECONE_INDEX_NAME not in [
                idx.name for idx in pc.list_indexes().indexes
            ]:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENVIRONMENT,
                    ),
                )
            index = pc.Index(PINECONE_INDEX_NAME)
            logger.info(
                f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}"
            )
            return index
        except Exception as e:
            logger.error("An error occurred while setting up Pinecone: %s", str(e))
            return None

    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer with initial data."""
        initial_data = ["example text for vectorizer fitting"]
        self.vectorizer.fit(initial_data)

    def update_memory(self, user_input: str, response: str):
        """
        Update the memory with the latest interaction.

        Args:
            user_input (str): The user's input.
            response (str): The chatbot's response.
        """
        # Update short-term memory
        self.short_term_memory.append({"user_input": user_input, "response": response})
        if (
            len(self.short_term_memory) > 10
        ):  # Limit short-term memory to last 10 interactions
            self.short_term_memory.pop(0)

        # Index the interaction in Pinecone and MongoDB for long-term memory
        if self.pinecone_index is not None:
            self.index_in_pinecone(user_input, response)
        if self.mongo_collection is not None:
            self.store_in_mongodb(user_input, response)

    def index_in_pinecone(self, user_input: str, response: str):
        """
        Index the interaction in Pinecone for fast retrieval.

        Args:
            user_input (str): The user's input.
            response (str): The chatbot's response.
        """
        try:
            vector = self.vectorizer.transform([user_input]).toarray()[0].tolist()
            id = str(datetime.now().timestamp())
            self.pinecone_index.upsert(vectors=[(id, vector, {"response": response})])
            logger.info("Indexed interaction in Pinecone")
        except Exception as e:
            logger.error("An error occurred while indexing in Pinecone: %s", str(e))

    def store_in_mongodb(self, user_input: str, response: str):
        """
        Store the interaction in MongoDB for long-term memory.

        Args:
            user_input (str): The user's input.
            response (str): The chatbot's response.
        """
        try:
            document = {
                "user_input": user_input,
                "response": response,
                "timestamp": datetime.now(),
                "expiry": datetime.now() + timedelta(days=90),
            }
            self.mongo_collection.insert_one(document)
            logger.info("Stored interaction in MongoDB")
        except pymongo_errors.PyMongoError as e:
            logger.error(
                "An error occurred while storing interaction in MongoDB: %s", str(e)
            )

    def get_relevant_context(self, query: str) -> Optional[str]:
        """
        Retrieve relevant context from memory based on the query.

        Args:
            query (str): The user's query.

        Returns:
            Optional[str]: The most relevant context, if found.
        """
        # Search in short-term memory first
        for interaction in reversed(self.short_term_memory):
            if query in interaction["user_input"]:
                return interaction["response"]

        # If not found, search in Pinecone and MongoDB
        if self.pinecone_index is not None:
            pinecone_results = self._search_pinecone(query)
            if pinecone_results is not None:
                return pinecone_results

        if self.mongo_collection is not None:
            mongo_results = self._search_mongodb(query)
            if mongo_results is not None:
                return mongo_results

        return None

    def _search_pinecone(self, query: str) -> Optional[str]:
        """
        Search for relevant context in Pinecone.

        Args:
            query (str): The user's query.

        Returns:
            Optional[str]: The most relevant response, if found.
        """
        try:
            vector = self.vectorizer.transform([query]).toarray()[0].tolist()
            results = self.pinecone_index.query(
                vector=vector, top_k=1, include_metadata=True
            )
            if results.matches:
                return results.matches[0].metadata["response"]
        except Exception as e:
            logger.error("An error occurred while searching Pinecone: %s", str(e))
        return None

    def _search_mongodb(self, query: str) -> Optional[str]:
        """
        Search for relevant context in MongoDB.

        Args:
            query (str): The user's query.

        Returns:
            Optional[str]: The most relevant response, if found.
        """
        try:
            documents = (
                self.mongo_collection.find(
                    {"$text": {"$search": query}}, {"score": {"$meta": "textScore"}}
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(1)
            )
            for doc in documents:
                return doc["response"]
        except pymongo_errors.PyMongoError as e:
            logger.error("An error occurred while searching MongoDB: %s", str(e))
        return None

    def cleanup_memory(self):
        """Clean up expired memory from MongoDB."""
        try:
            self.mongo_collection.delete_many({"expiry": {"$lt": datetime.now()}})
            logger.info("Cleaned up expired memory from MongoDB")
        except pymongo_errors.PyMongoError as e:
            logger.error("An error occurred while cleaning up MongoDB: %s", str(e))

    def delete_memory_by_query(self, query: str):
        """
        Delete specific memory entries based on a query.

        Args:
            query (str): The query to match the memory entries for deletion.
        """
        try:
            self.mongo_collection.delete_many(
                {"user_input": {"$regex": query, "$options": "i"}}
            )
            logger.info("Deleted memory entries matching query: %s", query)
        except pymongo_errors.PyMongoError as e:
            logger.error("An error occurred while deleting memory by query: %s", str(e))

    def get_all_memory(self) -> List[Dict[str, str]]:
        """
        Retrieve all stored memories from MongoDB.

        Returns:
            List[Dict[str, str]]: A list of all memories.
        """
        try:
            documents = self.mongo_collection.find({})
            all_memories = [
                {
                    "user_input": doc["user_input"],
                    "response": doc["response"],
                    "timestamp": doc["timestamp"],
                }
                for doc in documents
            ]
            logger.info("Retrieved all memory entries")
            return all_memories
        except pymongo_errors.PyMongoError as e:
            logger.error(
                "An error occurred while retrieving all memory entries: %s", str(e)
            )
            return []

    def get_memory_count(self) -> int:
        """
        Get the count of all stored memory entries in MongoDB.

        Returns:
            int: The count of memory entries.
        """
        try:
            count = self.mongo_collection.count_documents({})
            logger.info("Memory count retrieved: %d", count)
            return count
        except pymongo_errors.PyMongoError as e:
            logger.error("An error occurred while retrieving memory count: %s", str(e))
            return 0


# Instantiate the MemoryManager
memory_manager = MemoryManager()
