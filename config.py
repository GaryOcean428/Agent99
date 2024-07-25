"""
Configuration settings for the Chat99 project.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANYSCALE_API_KEY = os.getenv("ANYSCALE_API_KEY")

# Model Settings
DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
LOCAL_MODEL = "llama3.1:8b"

# Router Settings
DEFAULT_ROUTER = "mf"
DEFAULT_THRESHOLD = 0.11593

# Memory Settings
MAX_SHORT_TERM_MEMORY = 10
LONG_TERM_MEMORY_EXPIRY_DAYS = 7

# Logging
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ollama Settings
OLLAMA_API_BASE = "http://localhost:11434"
