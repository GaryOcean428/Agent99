# config.py

import os
import json

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "Warning: dotenv module not found. Make sure to set environment variables manually."
    )

# Load calibration results
CALIBRATION_FILE = "calibration_results.json"
DEFAULT_THRESHOLD = 0.5

if os.path.exists(CALIBRATION_FILE):
    try:
        with open(CALIBRATION_FILE, "r") as f:
            calibration_results = json.load(f)
        ROUTER_THRESHOLD = calibration_results.get(
            "optimal_threshold", DEFAULT_THRESHOLD
        )
    except json.JSONDecodeError:
        print(
            f"Warning: {CALIBRATION_FILE} is not a valid JSON file. Using default threshold."
        )
        ROUTER_THRESHOLD = DEFAULT_THRESHOLD
    except Exception as e:
        print(
            f"Warning: Error reading {CALIBRATION_FILE}. Using default threshold. Error: {str(e)}"
        )
        ROUTER_THRESHOLD = DEFAULT_THRESHOLD
else:
    print(f"Info: {CALIBRATION_FILE} not found. Using default threshold.")
    ROUTER_THRESHOLD = DEFAULT_THRESHOLD

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
DEFAULT_ROUTER = "advanced"

# Memory Settings
MAX_SHORT_TERM_MEMORY = 10
LONG_TERM_MEMORY_EXPIRY_DAYS = 7

# MongoDB Settings
MONGODB_URI = os.getenv("MONGO_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "chat99")

# Redis Settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Logging
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Groq Settings
GROQ_API_BASE = "https://api.groq.com/openai/v1"
