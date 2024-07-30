"""
Configuration settings for the Chat99 project.
"""

import os
from dotenv import load_dotenv

# Default values
DEFAULT_ROUTER = "mf"
DEFAULT_THRESHOLD = 0.11593
MAX_SHORT_TERM_MEMORY = 10
LONG_TERM_MEMORY_EXPIRY_DAYS = 7
DEBUG = False
LOG_LEVEL = "INFO"

# Model names
HIGH_TIER_MODEL = "claude-3-5-sonnet-20240620"
MID_TIER_MODEL = "llama-3.1-70b-versatile"
LOW_TIER_MODEL = "llama-3.1-8b-instant"

# API keys
OPENAI_API_KEY = None
ANTHROPIC_API_KEY = None
ANYSCALE_API_KEY = None
GROQ_API_KEY = None

# Environment variables

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANYSCALE_API_KEY = os.getenv("ANYSCALE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Settings
HIGH_TIER_MODEL = "claude-3-5-sonnet-20240620"
MID_TIER_MODEL = "llama-3.1-70b-versatile"
LOW_TIER_MODEL = "llama-3.1-8b-instant"

# Router Settings
DEFAULT_ROUTER = "mf"
DEFAULT_THRESHOLD = 0.11593

# Memory Settings
MAX_SHORT_TERM_MEMORY = 10
LONG_TERM_MEMORY_EXPIRY_DAYS = 7

# Logging
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Groq Settings
GROQ_API_BASE = "https://api.groq.com/openai/v1"
