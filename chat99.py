"""
chat99.py: Core functionality for Chat99 - An intelligent AI assistant with advanced memory
and multi-model capabilities.
"""

import os
import argparse
from typing import List, Dict, Tuple

# Import required libraries with error handling
try:
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from anthropic import Anthropic
    from groq import Groq
    from routellm.controller import Controller
    from config import (
        HIGH_TIER_MODEL, MID_TIER_MODEL, LOW_TIER_MODEL,
        DEFAULT_ROUTER, DEFAULT_THRESHOLD, MAX_SHORT_TERM_MEMORY
    )
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please make sure all required libraries are installed.")
    exit(1)

# Load environment variables
load_dotenv()

# Set up Rich console
console = Console()

def generate_response(model: str, conversation: List[Dict[str, str]], max_tokens: int = 1024) -> str:
    """Generate a response using the specified model."""
    try:
        if model == HIGH_TIER_MODEL:
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=conversation
            )
            return response.content[0].text
        elif model in [MID_TIER_MODEL, LOW_TIER_MODEL]:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            return "Error: Invalid model specified"
    except (Anthropic.APIError, Groq.APIError) as e:
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
    console.print(Panel("Welcome to Chat99 with Multi-Model Routing!",
                        title="Chat Interface", border_style="bold magenta"))

    if args.use_dynamic_routing:
        console.print(f"[bold yellow]Dynamic routing enabled using {args.router} router "
                      f"with threshold {args.threshold}[/bold yellow]")
    else:
        console.print("[bold yellow]Using tiered model selection based on input complexity.[/bold yellow]")
    console.print("Type 'exit' to end the conversation.")

    conversation: List[Dict[str, str]] = []
    controller = Controller(
        routers=[DEFAULT_ROUTER],
        strong_model=HIGH_TIER_MODEL,
        weak_model=MID_TIER_MODEL,
    )

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ").strip()

        if user_input.lower() == 'exit':
            console.print("[bold green]Chat99:[/bold green] Goodbye! It was nice chatting with you.")
            break

        conversation.append({"role": "user", "content": user_input})

        try:
            if args.use_dynamic_routing:
                response = controller.chat.completions.create(
                    model=f"router-{args.router}-{args.threshold}",
                    messages=conversation
                )
                model_used = response.model
                content = response.choices[0].message.content
                routing_explanation = f"Routed using {args.router} router with threshold {args.threshold}"
            else:
                complexity = "high" if len(user_input.split()) > 50 else "mid" if len(user_input.split()) > 20 else "low"
                model = HIGH_TIER_MODEL if complexity == "high" else MID_TIER_MODEL if complexity == "mid" else LOW_TIER_MODEL
                content = generate_response(model, conversation)
                model_used = model
                routing_explanation = f"Selected {model} based on input complexity: {complexity}"

            if content:
                console.print(f"[bold green]Chat99 ([italic]{model_used}[/italic]):[/bold green] ", end="")
                console.print(content)
                console.print(f"[bold yellow]Model Selection: {routing_explanation}[/bold yellow]")

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
        parser.add_argument("--use-dynamic-routing", action="store_true", help="Use dynamic routing")
        parser.add_argument("--router", type=str, default=DEFAULT_ROUTER,
                            help=f"Router to use (default: {DEFAULT_ROUTER})")
        parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                            help=f"Routing threshold (default: {DEFAULT_THRESHOLD})")
        chat_args = parser.parse_args()
        chat_with_99(chat_args)