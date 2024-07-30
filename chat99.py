"""
chat99.py: Core functionality for Chat99 - An intelligent AI assistant with advanced memory and multi-model capabilities.
"""

import os
import argparse
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from anthropic import Anthropic
from groq import Groq
from routellm.controller import Controller

# Load environment variables
load_dotenv()

# Import configuration
from config import (
    HIGH_TIER_MODEL,
    MID_TIER_MODEL,
    LOW_TIER_MODEL,
    DEFAULT_ROUTER,
    DEFAULT_THRESHOLD,
    MAX_SHORT_TERM_MEMORY
)

# Set up Rich console
console = Console()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chat99 - An intelligent AI assistant")
    parser.add_argument("--use-dynamic-routing", action="store_true", help="Use dynamic routing")
    parser.add_argument("--router", type=str, default=DEFAULT_ROUTER, help=f"Router to use (default: {DEFAULT_ROUTER})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Routing threshold (default: {DEFAULT_THRESHOLD})")
    return parser.parse_args()

def setup_routellm_controller() -> Controller:
    """Set up and return a RouteLLM controller."""
    return Controller(
        routers=[DEFAULT_ROUTER],
        strong_model=HIGH_TIER_MODEL,
        weak_model=MID_TIER_MODEL,
    )

def generate_response(
    controller: Controller,
    conversation: List[Dict[str, str]],
    router: str,
    threshold: float
) -> Tuple[str, str, str]:
    """Generate a response using RouteLLM for routing."""
    try:
        response = controller.chat.completions.create(
            model=f"router-{router}-{threshold}",
            messages=conversation
        )
        model_used = response.model  # RouteLLM will update this to indicate which model was actually used
        content = response.choices[0].message.content
        return content, model_used, f"Routed using {router} router with threshold {threshold}"
    except Exception as e:
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
        return None, None, None

def determine_complexity(user_input: str) -> str:
    """Determine the complexity of the user input to choose the appropriate model."""
    # This is a simple heuristic and can be improved with more sophisticated NLP techniques
    if len(user_input.split()) > 50 or "code" in user_input.lower() or "complex" in user_input.lower():
        return "high"
    elif len(user_input.split()) > 20:
        return "mid"
    else:
        return "low"

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
        console.print(f"[bold yellow]Dynamic routing enabled using {args.router} router with threshold {args.threshold}[/bold yellow]")
    else:
        console.print("[bold yellow]Using tiered model selection based on input complexity.[/bold yellow]")
    console.print("Type 'exit' to end the conversation.")

    conversation: List[Dict[str, str]] = []
    controller = setup_routellm_controller()
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ").strip()

        if user_input.lower() == 'exit':
            console.print("[bold green]Chat99:[/bold green] Goodbye! It was nice chatting with you.")
            break

        conversation.append({"role": "user", "content": user_input})

        if args.use_dynamic_routing:
            response, model_used, routing_explanation = generate_response(controller, conversation, args.router, args.threshold)
        else:
            complexity = determine_complexity(user_input)
            if complexity == "high":
                response = anthropic_client.messages.create(
                    model=HIGH_TIER_MODEL,
                    max_tokens=1024,
                    messages=conversation
                ).content[0].text
                model_used = HIGH_TIER_MODEL
            else:
                groq_model = MID_TIER_MODEL if complexity == "mid" else LOW_TIER_MODEL
                response = groq_client.chat.completions.create(
                    model=groq_model,
                    messages=conversation,
                    max_tokens=1024
                ).choices[0].message.content
                model_used = groq_model
            routing_explanation = f"Selected {model_used} based on input complexity: {complexity}"

        if response:
            console.print(f"[bold green]Chat99 ([italic]{model_used}[/italic]):[/bold green] ", end="")
            console.print(response)
            console.print(f"[bold yellow]Model Selection: {routing_explanation}[/bold yellow]")

            conversation.append({"role": "assistant", "content": response})

            if len(conversation) > MAX_SHORT_TERM_MEMORY:
                conversation = conversation[-MAX_SHORT_TERM_MEMORY:]

            display_message("assistant", response)
        else:
            console.print("[bold red]Failed to get a response. Please try again.[/bold red]")

def check_api_keys() -> bool:
    """Check if the required API keys are set."""
    required_keys = ["ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        console.print(f"[bold red]Error: The following API keys are not set: {', '.join(missing_keys)}[/bold red]")
        console.print("Please make sure your API keys are correctly set in the .env file.")
        return False
    return True

if __name__ == "__main__":
    if check_api_keys():
        chat_args = parse_arguments()
        chat_with_99(chat_args)