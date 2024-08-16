"""
Configuration settings for the Chat99 project.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Model Settings
HIGH_TIER_MODEL = "claude-3-5-sonnet-20240620"
MID_TIER_MODEL = "llama-3.1-70b-versatile"
LOW_TIER_MODEL = "llama-3.1-8b-instant"

# Router Settings
ROUTER_THRESHOLD = float(os.getenv("ROUTER_THRESHOLD", "0.7"))

# Memory Settings
MAX_SHORT_TERM_MEMORY = int(os.getenv("MAX_SHORT_TERM_MEMORY", "10"))
LONG_TERM_MEMORY_EXPIRY_DAYS = int(os.getenv("LONG_TERM_MEMORY_EXPIRY_DAYS", "90"))

# MongoDB Settings
MONGO_URI = os.getenv("MONGO_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "chat99")

# Redis Settings
REDIS_HOST = os.getenv("REDIS_HOST", "real-wren-52199.upstash.io")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Search Settings
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# Logging
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Application Settings
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))
CONFIGURE_THRESHOLD = float(os.getenv("CONFIGURE_THRESHOLD", "0.5"))

# Pinecone Settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_DIMENSION = 128  # Make sure this matches your Pinecone index dimension

# Add any other configuration settings as needed
