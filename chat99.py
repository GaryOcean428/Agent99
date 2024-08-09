"""
chat99.py: Main module for the Chat99 AI assistant.

This module integrates various components including advanced routing,
memory management, retrieval-augmented generation (RAG), and web search
to provide a sophisticated chatbot experience. It supports multiple
language models and adaptive response strategies.
"""

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from anthropic import Anthropic
from groq import Groq
from rich.console import Console
from advanced_router import advanced_router
from memory_manager import memory_manager
from search import perform_search
from rag import retrieve_relevant_info

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up Rich console
console = Console()

# Define model constants
HIGH_TIER_MODEL = "claude-3-5-sonnet-20240620"
MID_TIER_MODEL = "llama-3.1-70b-versatile"
LOW_TIER_MODEL = "llama-3.1-8b-instant"

def generate_response(
    model: str,
    conversation: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    response_strategy: str = "default",
) -> str:
    """
    Generate a response using the specified model and parameters.

    Args:
        model (str): The model to use for generating the response.
        conversation (List[Dict[str, str]]): The conversation history.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for response generation.
        response_strategy (str): The strategy to use for generating the response.

    Returns:
        str: The generated response.
    """
    try:
        # Get relevant context from memory and RAG
        context = memory_manager.get_relevant_context(conversation[-1]["content"]) if memory_manager else ""
        rag_info = retrieve_relevant_info(conversation[-1]["content"])
        search_results = perform_search(conversation[-1]["content"])

        # Prepare messages for API call
        messages = conversation.copy()
        if context or rag_info or search_results:
            system_message = (
                f"Relevant context: {context}\n\n"
                f"Retrieved information: {rag_info}\n\n"
                f"Search results: {search_results}"
            )
            messages.insert(0, {"role": "system", "content": system_message})

        # Add response strategy instruction
        strategy_instruction = get_strategy_instruction(response_strategy)
        messages.insert(0, {"role": "system", "content": strategy_instruction})

        # Generate response using the appropriate model
        if model == HIGH_TIER_MODEL:
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            api_response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            generated_response = response.content[0].text
        elif model in [MID_TIER_MODEL, LOW_TIER_MODEL]:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            generated_response = response.choices[0].message.content
        else:
            raise ValueError(f"Invalid model specified: {model}")

        # Update memory with the new interaction
        if memory_manager:
            memory_manager.update_memory(conversation[-1]["content"], generated_response)

        return generated_response

    except Exception as e:
        logger.error("An error occurred during response generation: %s", str(e))
        return "I apologize, but I encountered an error while processing your request. Please try again later or contact support if the issue persists."

def get_strategy_instruction(strategy: str) -> str:
    """
    Get the instruction for the specified response strategy.

    Args:
        strategy (str): The response strategy to use.

    Returns:
        str: The instruction for the specified strategy.
    """
    strategies = {
        "casual_conversation": "Respond in a casual, friendly manner without using any specific format.",
        "chain_of_thought": "Use a step-by-step reasoning approach. Break down the problem, consider relevant information, and explain your thought process clearly.",
        "direct_answer": "Provide a concise, direct answer to the question without unnecessary elaboration.",
        "boolean_with_explanation": "Start with a clear Yes or No, then provide a brief explanation for your answer.",
    }
    return strategies.get(strategy, "Respond naturally to the query, providing relevant information and insights.")

def chat_with_99(user_input: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """
    Process user input and generate a response using the Chat99 system.

    Args:
        user_input (str): The user's input message.
        conversation_history (List[Dict[str, str]], optional): The conversation history.

    Returns:
        str: The generated response from the Chat99 system.
    """
    if conversation_history is None:
        conversation_history = []

    conversation_history.append({"role": "user", "content": user_input})

    try:
        route_config = advanced_router.route(user_input, conversation_history)
        model = route_config["model"]
        max_tokens = route_config["max_tokens"]
        temperature = route_config["temperature"]
        response_strategy = route_config["response_strategy"]

        logger.info("Routing decision: %s", route_config)

        response = generate_response(
            model, conversation_history, max_tokens, temperature, response_strategy
        )

        conversation_history.append({"role": "assistant", "content": response})

        return response
    except Exception as e:
        logger.error("An error occurred in chat_with_99: %s", str(e))
        return "I'm sorry, but an error occurred while processing your request. Please try again later."

if __name__ == "__main__":
    print("Welcome to Chat99! Type 'exit' to end the conversation.")
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chat99: Goodbye! It was nice chatting with you.")
            break

        response = chat_with_99(user_input, conversation_history)
        print(f"Chat99: {response}")