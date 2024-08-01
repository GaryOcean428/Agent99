import os
import argparse
from typing import List, Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from anthropic import Anthropic
from groq import Groq

from config import HIGH_TIER_MODEL, MID_TIER_MODEL, LOW_TIER_MODEL
from advanced_router import advanced_router
from irac_framework import apply_irac_framework, apply_comparative_analysis
from memory_manager import MemoryManager

# Load environment variables
load_dotenv()

# Set up Rich console
console = Console()

# Initialize MemoryManager
try:
    memory_manager = MemoryManager()
except Exception as e:
    console.print(f"[bold red]Error initializing MemoryManager: {str(e)}[/bold red]")
    console.print("[bold yellow]Continuing without long-term memory functionality.[/bold yellow]")
    memory_manager = None

def generate_response(model: str, conversation: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7, response_strategy: str = "default") -> str:
    try:
        # Get relevant context from memory
        context = ""
        if memory_manager:
            context = memory_manager.get_relevant_context(conversation[-1]['content'])
        
        # Add context to the conversation
        if context:
            conversation.insert(-1, {"role": "system", "content": f"Relevant context: {context}"})

        if model == HIGH_TIER_MODEL:
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=conversation
            )
            raw_response = response.content[0].text
        elif model in [MID_TIER_MODEL, LOW_TIER_MODEL]:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                max_tokens=max_tokens,
                temperature=temperature
            )
            raw_response = response.choices[0].message.content
        else:
            return "Error: Invalid model specified"

        # Apply the appropriate response strategy
        if response_strategy == "irac":
            processed_response = apply_irac_framework(conversation[-1]['content'], raw_response)
        elif response_strategy == "direct_answer":
            processed_response = f"Direct Answer: {raw_response}"
        elif response_strategy == "boolean_with_explanation":
            processed_response = f"Yes/No: {'Yes' if 'yes' in raw_response.lower() else 'No'}\nExplanation: {raw_response}"
        elif response_strategy == "comparative_analysis":
            processed_response = apply_comparative_analysis(conversation[-1]['content'], raw_response)
        else:
            processed_response = raw_response

        # Update memory with the new interaction
        if memory_manager:
            memory_manager.update_memory(conversation[-1]['content'], processed_response)

        return processed_response

    except Exception as e:
        console.print(f"[bold red]API Error: {str(e)}[/bold red]")
        return ""

def display_message(role: str, content: str) -> None:
    if role == "user":
        console.print(Panel(content, expand=False, border_style="blue", title="You"))
    else:
        md = Markdown(content)
        console.print(Panel(md, expand=False, border_style="green", title="Chat99"))

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

        # Check for cached response
        cached_response = memory_manager.get_cached_response(user_input)
        if cached_response:
            console.print(f"[bold green]Chat99 (Cached):[/bold green] {cached_response}")
            continue

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
            console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")

def check_api_keys() -> bool:
    required_keys = ["ANTHROPIC_API_KEY", "GROQ_API_KEY", "MONGO_DATA_API_KEY", "MONGO_URI", "REDIS_PASSWORD"]
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