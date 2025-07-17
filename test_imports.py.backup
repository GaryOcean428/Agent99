"""
Test imports for Agent99 components.

This module tests the imports of various components used in the Agent99 project.
It mocks necessary modules and initializes key components to ensure they can be
imported and used without errors.
"""

from config import Config
from memory_manager import MemoryManager
from response_generator import ResponseGenerator
from input_analyzer import InputAnalyzer
import sys
from unittest.mock import Mock

# Mock required modules
sys.modules["anthropic"] = Mock()
sys.modules["groq"] = Mock()

# Import components

print(f"Python version: {sys.version}")

print("Initializing components...")
config = Config()
memory_manager = MemoryManager()
input_analyzer = InputAnalyzer(config)

# Mock the API clients
response_generator = ResponseGenerator(config, memory_manager)
response_generator.model_manager._call_anthropic_api = Mock(
    return_value="Mocked Anthropic response"
)
response_generator.model_manager._call_groq_api = Mock(
    return_value="Mocked Groq response"
)

print("Testing response generation...")
TEST_INPUT = "What is the capital of France?"
response = response_generator.generate(TEST_INPUT)
print(f"Input: {TEST_INPUT}")
print(f"Response: {response}")

print("All imports and initializations successful!")
