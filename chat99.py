"""
chat99.py: Core functionality for Chat99 - An intelligent AI assistant with advanced memory
and multi-model capabilities.
"""

import os
import argparse
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from anthropic import Anthropic
from groq import Groq

from config import HIGH_TIER_MODEL, MID_TIER_MODEL, LOW_TIER_MODEL, MAX_SHORT_TERM_MEMORY
from advanced_router import advanced_router
from irac_framework import apply_irac_framework, apply_comparative_analysis

# Load environment variables
load_dotenv()

# Set up Rich console
console = Console()

def generate_response(model: str, conversation: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7, response_strategy: str = "default") -> str:
    """Generate a response using the specified model and apply the appropriate response strategy."""
    try:
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
            return apply_irac_framework(conversation[-1]['content'], raw_response)
        elif response_strategy == "direct_answer":
            return f"Direct Answer: {raw_response}"
        elif response_strategy == "boolean_with_explanation":
            return f"Yes/No: {'Yes' if 'yes' in raw_response.lower() else 'No'}\nExplanation: {raw_response}"
        elif response_strategy == "comparative_analysis":
            return apply_comparative_analysis(conversation[-1]['content'], raw_response)
        else:
            return raw_response

    except Exception as e:
        console.print(f"[bold red]API Error: {str(e)}[/bold red]")
        return ""

def display_message(role: str, content: str) -> None:
    """Display a message in the chat interface."""
    if role == "user":
        console.print(Panel(content, expand=False, border_style="blue", title="You"))
    else:
        md = Markdown(content)
        console.print(Panel(md, expand=False, border_style="green", title="Chat99"))

def chat_with_99(args: argparse.Namespace) -> None:
    """Main chat loop for interacting with the AI."""
    console.print(Panel("Welcome to Chat99 with Advanced Routing and Dynamic Response Strategies!", 
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

                if len(conversation) > MAX_SHORT_TERM_MEMORY:
                    conversation = conversation[-MAX_SHORT_TERM_MEMORY:]

                display_message("assistant", content)
            else:
                console.print("[bold red]Failed to get a response. Please try again.[/bold red]")
        except Exception as e:
            console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")

def check_api_keys() -> bool:
    """Check if the required API keys are set."""
    required_keys = ["ANTHROPIC_API_KEY", "GROQ_API_KEY"]
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