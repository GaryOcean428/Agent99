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
from advanced_router import advanced_router
from memory_manager import memory_manager
from search import perform_search
from rag import retrieve_relevant_info

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    try:
        # Get relevant context from memory and RAG
        context = ""
        if memory_manager:
            context = memory_manager.get_relevant_context(conversation[-1]["content"])

        rag_info = retrieve_relevant_info(conversation[-1]["content"])

        # Perform web search
        search_results = perform_search(conversation[-1]["content"])

        # Prepare messages for API call
        messages = conversation.copy()
        if context or rag_info or search_results:
            system_message = f"Relevant context: {context}\n\nRetrieved information: {rag_info}\n\nSearch results: {search_results}"
            messages.insert(0, {"role": "system", "content": system_message})

        # Add response strategy instruction
        strategy_instruction = get_strategy_instruction(response_strategy)
        messages.insert(0, {"role": "system", "content": strategy_instruction})

        # Generate response using the appropriate model
        if model == HIGH_TIER_MODEL:
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                messages=messages,
            )
            return response.content[0].text
        elif model in [MID_TIER_MODEL, LOW_TIER_MODEL]:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                temperature=temperature,
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Invalid model specified: {model}")

        # Update memory with the new interaction
        if memory_manager:
            memory_manager.update_memory(conversation[-1]["content"], raw_response)

        return raw_response

    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Network error: {str(e)}[/bold red]")
    except Exception as e:
        logger.error(f"An error occurred during response generation: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again later or contact support if the issue persists."


def get_strategy_instruction(strategy: str) -> str:
    if strategy == "casual_conversation":
        return "Respond in a casual, friendly manner without using any specific format."
    elif strategy == "chain_of_thought":
        return "Use a step-by-step reasoning approach. Break down the problem, consider relevant information, and explain your thought process clearly."
    elif strategy == "direct_answer":
        return "Provide a concise, direct answer to the question without unnecessary elaboration."
    elif strategy == "boolean_with_explanation":
        return "Start with a clear Yes or No, then provide a brief explanation for your answer."
    else:
        return "Respond naturally to the query, providing relevant information and insights."


def chat_with_99(
    user_input: str, conversation_history: List[Dict[str, str]] = None
) -> str:
    if conversation_history is None:
        conversation_history = []

    conversation_history.append({"role": "user", "content": user_input})

    try:
        route_config = advanced_router.route(user_input, conversation_history)
        model = route_config["model"]
        max_tokens = route_config["max_tokens"]
        temperature = route_config["temperature"]
        response_strategy = route_config["response_strategy"]

        logger.info(f"Routing decision: {route_config}")

        response = generate_response(
            model, conversation_history, max_tokens, temperature, response_strategy
        )

        conversation_history.append({"role": "assistant", "content": response})

        return response
    except Exception as e:
        logger.error(f"An error occurred in chat_with_99: {str(e)}")
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
