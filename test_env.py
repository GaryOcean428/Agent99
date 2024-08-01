# test_env.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("MONGO_URI:", os.getenv("MONGO_URI"))
print("MONGODB_DB_NAME:", os.getenv("MONGODB_DB_NAME"))
print("REDIS_HOST:", os.getenv("REDIS_HOST"))
print("REDIS_PORT:", os.getenv("REDIS_PORT"))
print("REDIS_DB:", os.getenv("REDIS_DB"))
