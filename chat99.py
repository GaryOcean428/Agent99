import os
import argparse
from typing import List, Dict
import time
import logging

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from anthropic import Anthropic
from groq import Groq

from config import HIGH_TIER_MODEL, MID_TIER_MODEL, LOW_TIER_MODEL
from advanced_router import advanced_router
from memory_manager import MemoryManager
from search import perform_search
from rag import retrieve_relevant_info

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up Rich console
console = Console()

# Initialize MemoryManager
try:
    memory_manager = MemoryManager()
except Exception as e:
    logger.error(f"Error initializing MemoryManager: {str(e)}")
    console.print("[bold red]Error initializing MemoryManager. Continuing without memory functionality.[/bold red]")
    memory_manager = None

def generate_response(model: str, conversation: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7, response_strategy: str = "default") -> str:
    try:
        # Get relevant context from memory and RAG
        context = ""
        if memory_manager:
            context = memory_manager.get_relevant_context(conversation[-1]['content'])
        
        rag_info = retrieve_relevant_info(conversation[-1]['content'])
        
        # Perform web search
        search_results = perform_search(conversation[-1]['content'])

        # Prepare messages for API call
        messages = conversation.copy()
        if context or rag_info or search_results:
            system_message = f"Relevant context: {context}\n\nRetrieved information: {rag_info}\n\nSearch results: {search_results}"
            messages.insert(0, {"role": "system", "content": system_message})

        # Add response strategy instruction
        strategy_instruction = get_strategy_instruction(response_strategy)
        messages.insert(0, {"role": "system", "content": strategy_instruction})

        start_time = time.time()

        if model == HIGH_TIER_MODEL:
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            raw_response = response.content[0].text
        elif model in [MID_TIER_MODEL, LOW_TIER_MODEL]:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            raw_response = response.choices[0].message.content
        else:
            raise ValueError(f"Invalid model specified: {model}")

        end_time = time.time()
        logger.info(f"Response time: {end_time - start_time:.2f} seconds")

        # Update memory with the new interaction
        if memory_manager:
            memory_manager.update_memory(conversation[-1]['content'], raw_response)

        return raw_response

    except Exception as e:
        logger.error(f"An error occurred during response generation: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Could you please try again?"

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

def chat_with_99(args: argparse.Namespace) -> None:
    console.print(Panel("Welcome to Chat99 with Advanced Routing, Dynamic Response Strategies, and Enhanced Memory!", 
                        title="Chat Interface", border_style="bold magenta"))
    console.print("Type 'exit' to end the conversation.")

    conversation: List[Dict[str, str]] = []

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ").strip()

        if user_input.lower() == 'exit':
            console.print("[bold green]Chat99:[/bold green] Goodbye! It was nice chatting with you.")
            break

        conversation.append({"role": "user", "content": user_input})

        try:
            route_config = advanced_router.route(user_input, conversation)
            model = route_config['model']
            max_tokens = route_config['max_tokens']
            temperature = route_config['temperature']
            response_strategy = route_config['response_strategy']

            content = generate_response(model, conversation, max_tokens, temperature, response_strategy)

            if content:
                console.print(f"[bold green]Chat99 ([italic]{model}[/italic]):[/bold green] ", end="")
                console.print(content)
                console.print(f"[bold yellow]Model Selection: {route_config['routing_explanation']}[/bold yellow]")
                console.print(f"[bold yellow]Response Strategy: {response_strategy}[/bold yellow]")

                conversation.append({"role": "assistant", "content": content})

                display_message("assistant", content)
            else:
                console.print("[bold red]Failed to get a response. Please try again.[/bold red]")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            console.print(f"[bold red]An unexpected error occurred. Please try again.[/bold red]")

def display_message(role: str, content: str) -> None:
    if role == "user":
        console.print(Panel(content, expand=False, border_style="blue", title="You"))
    else:
        md = Markdown(content)
        console.print(Panel(md, expand=False, border_style="green", title="Chat99"))

def check_api_keys() -> bool:
    required_keys = ["ANTHROPIC_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY", "SERP_API_KEY", "MONGO_DATA_API_KEY", "MONGO_URI", "REDIS_PASSWORD"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        console.print(f"[bold red]Error: The following API keys are not set: {', '.join(missing_keys)}[/bold red]")
        console.print("Please make sure your API keys are correctly set in the .env file.")
        return False
    return True

if __name__ == "__main__":
    if check_api_keys():
        parser = argparse.ArgumentParser(description="Chat99 - An intelligent AI assistant")
        chat_args = parser.parse_args()
        chat_with_99(chat_args)