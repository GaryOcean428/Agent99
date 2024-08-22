"""
Configuration settings for the Chat99 project.
"""

import toml


def get_config(section, key, default=None, value_type=str):
    """
    Get a configuration value from the loaded config dictionary.

    Args:
        section (str): The section of the configuration.
        key (str): The key of the configuration item.
        default: The default value if the key is not found.
        value_type: The type to cast the value to.

    Returns:
        The configuration value, or None if not found and no default provided.
    """
    value = config.get(section, {}).get(key, default)
    if value is None:
        return None
    try:
        return value_type(value)
    except ValueError:
        print(f"Warning: Invalid value for {section}.{key}. Using default.")
        return default


# Load configuration from TOML file
config = {}
try:
    with open("config.toml", "r", encoding="utf-8") as f:
        config = toml.load(f)
except FileNotFoundError:
    print("Warning: config.toml not found. Using default values where possible.")

# API Keys
ANTHROPIC_API_KEY = get_config("api_keys", "ANTHROPIC_API_KEY")
GROQ_API_KEY = get_config("api_keys", "GROQ_API_KEY")
GOOGLE_API_KEY = get_config("api_keys", "GOOGLE_API_KEY")
GITHUB_TOKEN = get_config("api_keys", "GITHUB_TOKEN")

# Model Settings
HIGH_TIER_MODEL = get_config(
    "model_settings", "HIGH_TIER_MODEL", "claude-3-5-sonnet-20240620"
)
MID_TIER_MODEL = get_config(
    "model_settings", "MID_TIER_MODEL", "llama-3.1-70b-versatile"
)
LOW_TIER_MODEL = get_config("model_settings", "LOW_TIER_MODEL", "llama-3.1-8b-instant")

# Router Settings
ROUTER_THRESHOLD = get_config("router_settings", "ROUTER_THRESHOLD", 0.7, float)

# Memory Settings
MAX_SHORT_TERM_MEMORY = get_config("memory_settings", "MAX_SHORT_TERM_MEMORY", 10, int)
LONG_TERM_MEMORY_EXPIRY_DAYS = get_config(
    "memory_settings", "LONG_TERM_MEMORY_EXPIRY_DAYS", 90, int
)

# MongoDB Settings
MONGO_URI = get_config("mongodb_settings", "MONGO_URI")
MONGODB_DB_NAME = get_config("mongodb_settings", "MONGODB_DB_NAME", "chat99")

# Redis Settings
REDIS_HOST = get_config("redis_settings", "REDIS_HOST", "real-wren-52199.upstash.io")
REDIS_PORT = get_config("redis_settings", "REDIS_PORT", 6379, int)
REDIS_PASSWORD = get_config("redis_settings", "REDIS_PASSWORD")

# Search Settings
SEARCH_ENGINE_ID = get_config("search_settings", "SEARCH_ENGINE_ID")

# Logging
DEBUG = get_config("logging", "DEBUG", False, bool)
LOG_LEVEL = get_config("logging", "LOG_LEVEL", "INFO")

# Application Settings
MAX_CONVERSATION_HISTORY = get_config(
    "application_settings", "MAX_CONVERSATION_HISTORY", 50, int
)
CONFIGURE_THRESHOLD = get_config(
    "application_settings", "CONFIGURE_THRESHOLD", 0.5, float
)

# Pinecone Settings
PINECONE_API_KEY = get_config("pinecone_settings", "PINECONE_API_KEY")
PINECONE_ENVIRONMENT = get_config("pinecone_settings", "PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = get_config("pinecone_settings", "PINECONE_INDEX_NAME")
PINECONE_DIMENSION = get_config("pinecone_settings", "PINECONE_DIMENSION", 128, int)


class Config:
    """A class to hold all configuration settings."""

    def __init__(self):
        for key, value in globals().items():
            if key.isupper():
                setattr(self, key, value)


# Print warnings for missing critical configurations
if not ANTHROPIC_API_KEY:
    print("Warning: ANTHROPIC_API_KEY is not set. Some features may not work.")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY is not set. Some features may not work.")
if not MONGO_URI:
    print("Warning: MONGO_URI is not set. Database features may not work.")
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT or not PINECONE_INDEX_NAME:
    print(
        "Warning: Pinecone configuration is incomplete. "
        "Vector search features may not work."
    )
