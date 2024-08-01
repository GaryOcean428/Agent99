import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGODB_URI = os.getenv("MONGO_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "chat99")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Create a collection and insert a test document
collection = db["test_collection"]
test_document = {"message": "Hello, MongoDB!"}
collection.insert_one(test_document)

# Retrieve the test document
retrieved_document = collection.find_one({"message": "Hello, MongoDB!"})
print("Retrieved Document:", retrieved_document)
