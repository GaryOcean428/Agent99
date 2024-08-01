from memory_manager import MemoryManager
import os

def test_memory_manager():
    # Check if the API key is set
    if not os.getenv("MONGO_DATA_API_KEY"):
        print("Error: MONGO_DATA_API_KEY is not set in the environment variables.")
        return

    manager = MemoryManager()

    try:
        # Test Redis connection
        print("Testing Redis connection...")
        manager.redis_client.ping()
        print("Redis connection successful.")

        # Test adding to memory
        manager.update_memory("What is the capital of France?", "The capital of France is Paris.")
        print("Memory updated successfully.")
        
        # Test retrieving context
        context = manager.get_relevant_context("Tell me about France's capital")
        print("Retrieved context:", context)

        # Test caching
        cached_response = manager.get_cached_response("What is the capital of France?")
        print("Cached response:", cached_response)

        # Test complexity determination
        complexity = manager.determine_complexity("What is the impact of artificial intelligence on modern society?")
        print("Query complexity:", complexity)

        # Test cleanup (this won't delete anything if we just added the data)
        manager.cleanup_memory()
        print("Memory cleanup performed.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_memory_manager()