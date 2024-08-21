"""
Configuration settings for the Chat99 project.
"""

import toml

# Load configuration from TOML file
with open('config.toml', 'r') as f:
    config = toml.load(f)

# API Keys
ANTHROPIC_API_KEY = config.get('api_keys', {}).get('ANTHROPIC_API_KEY')
GROQ_API_KEY = config.get('api_keys', {}).get('GROQ_API_KEY')
GOOGLE_API_KEY = config.get('api_keys', {}).get('GOOGLE_API_KEY')
GITHUB_TOKEN = config.get('api_keys', {}).get('GITHUB_TOKEN')

# Model Settings
HIGH_TIER_MODEL = config.get('model_settings', {}).get('HIGH_TIER_MODEL', "claude-3-5-sonnet-20240620")
MID_TIER_MODEL = config.get('model_settings', {}).get('MID_TIER_MODEL', "llama-3.1-70b-versatile")
LOW_TIER_MODEL = config.get('model_settings', {}).get('LOW_TIER_MODEL', "llama-3.1-8b-instant")

# Router Settings
ROUTER_THRESHOLD = float(config.get('router_settings', {}).get('ROUTER_THRESHOLD', 0.7))

# Memory Settings
MAX_SHORT_TERM_MEMORY = int(config.get('memory_settings', {}).get('MAX_SHORT_TERM_MEMORY', 10))
LONG_TERM_MEMORY_EXPIRY_DAYS = int(config.get('memory_settings', {}).get('LONG_TERM_MEMORY_EXPIRY_DAYS', 90))

# MongoDB Settings
MONGO_URI = config.get('mongodb_settings', {}).get('MONGO_URI')
MONGODB_DB_NAME = config.get('mongodb_settings', {}).get('MONGODB_DB_NAME', "chat99")

# Redis Settings
REDIS_HOST = config.get('redis_settings', {}).get('REDIS_HOST', "real-wren-52199.upstash.io")
REDIS_PORT = int(config.get('redis_settings', {}).get('REDIS_PORT', 6379))
REDIS_PASSWORD = config.get('redis_settings', {}).get('REDIS_PASSWORD')

# Search Settings
SEARCH_ENGINE_ID = config.get('search_settings', {}).get('SEARCH_ENGINE_ID')

# Logging
DEBUG = config.get('logging', {}).get('DEBUG', "False").lower() == "true"
LOG_LEVEL = config.get('logging', {}).get('LOG_LEVEL', "INFO")

# Application Settings
MAX_CONVERSATION_HISTORY = int(config.get('application_settings', {}).get('MAX_CONVERSATION_HISTORY', 50))
CONFIGURE_THRESHOLD = float(config.get('application_settings', {}).get('CONFIGURE_THRESHOLD', 0.5))

# Pinecone Settings
PINECONE_API_KEY = config.get('pinecone_settings', {}).get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = config.get('pinecone_settings', {}).get('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = config.get('pinecone_settings', {}).get('PINECONE_INDEX_NAME')
PINECONE_DIMENSION = config.get('pinecone_settings', {}).get('PINECONE_DIMENSION', 128)

# Add any other configuration settings as needed

class Config:
    def __init__(self):
        for key, value in globals().items():
            if key.isupper():
                setattr(self, key, value)
