# test_memory.py

from memory_manager import MemoryManager
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_memory():
    memory_manager = MemoryManager()

    user_input = "What is the weather like today?"
    response = "The weather is sunny with a high of 75 degrees."

    # Update memory with user input and response
    memory_manager.update_memory(user_input, response)

    # Retrieve short-term memory
    short_term_memory = memory_manager.get_relevant_context(user_input)
    print(f"Short-term memory: {short_term_memory}")

    # Retrieve long-term memory
    long_term_memory = memory_manager.get_relevant_context(user_input)
    print(f"Long-term memory: {long_term_memory}")

    # Retrieve cached response
    cached_response = memory_manager.get_cached_response(user_input)
    print(f"Cached response: {cached_response}")


if __name__ == "__main__":
    test_memory()
