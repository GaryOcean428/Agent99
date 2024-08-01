import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGO_URI")

if not MONGODB_URI:
    print("Error: MONGO_URI environment variable is not set.")
    exit(1)

print(f"Attempting to connect with URI: {MONGODB_URI.split('@')[0]}@[REDACTED]")

try:
    client = MongoClient(MONGODB_URI)
    db = client.get_database()
    print(f"Successfully connected to database: {db.name}")
    collection_names = db.list_collection_names()
    print(f"Collections in this database: {collection_names}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'client' in locals():
        client.close()
        print("MongoDB connection closed.")