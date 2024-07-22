"""
Chat99 - An AI assistant powered by Claude models.
This script implements a chat interface for interacting with the AI.
"""

import os
import argparse
from typing import List, Dict, Any
from models import get_model_info, get_model_list

# Try to import required libraries, provide installation instructions if not found
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from anthropic import Anthropic
    import anthropic
except ImportError:
    print("Required libraries not found. Please install them using:")
    print("pip install rich anthropic")
    exit(1)

# Set up Rich console
console = Console()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chat99 (powered by Claude)")
    parser.add_argument(
        "--max-tokens", type=int, default=2000, help="Maximum tokens for response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation",
    )
    return parser.parse_args()


def display_message(role: str, content: str) -> None:
    """Display a message in the chat interface."""
    if role == "user":
        console.print(Panel(content, expand=False, border_style="blue", title="You"))
    else:
        md = Markdown(content)
        console.print(Panel(md, expand=False, border_style="green", title="Chat99"))


def generate_response(
    client: Any,
    model_name: str,
    system_prompt: str,
    conversation: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate a response from the AI model."""
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=conversation,
        )
        return response.content[0].text
    except (anthropic.APIError, anthropic.APIConnectionError) as e:
        console.print(f"[bold red]An API error occurred: {str(e)}[/bold red]")
    except anthropic.AuthenticationError:
        console.print(
            "[bold red]Authentication error: Please check your API key.[/bold red]"
        )
    except anthropic.RateLimitError:
        console.print(
            "[bold red]Rate limit exceeded: Please try again later.[/bold red]"
        )
    except anthropic.APIStatusError as e:
        console.print(f"[bold red]API status error: {str(e)}[/bold red]")
    except KeyError:
        console.print("[bold red]Unexpected response format from the API.[/bold red]")
    except ValueError:
        console.print("[bold red]Invalid value in the API response.[/bold red]")
    return ""


def chat_with_99(args: argparse.Namespace) -> None:
    """Main chat loop for interacting with the AI."""
    console.print(
        Panel(
            "Welcome to Chat99 (powered by Claude)!",
            title="Chat Interface",
            border_style="bold magenta",
        )
    )

    console.print("Available models:")
    for key, model in get_model_list().items():
        console.print(f"{key}. {model['name']}")

    preferred_model = Prompt.ask(
        "Enter the number of your preferred model",
        choices=list(get_model_list().keys()),
    )
    model_info = get_model_info(preferred_model)

    model_name = model_info["id"]
    chat99_version = f"Chat99 {model_info['name'].title()}"
    console.print(f"[bold green]Using {chat99_version}[/bold green]")
    console.print("Type 'exit' to end the conversation, 'switch' to change chat mode.")

    general_prompt = (
        f"You are Chat99, an AI assistant powered by Claude {model_info['name'].title()}. "
        "You are helpful, honest, and harmless. You have extensive knowledge in various "
        "fields and can engage in conversations on a wide range of topics. You provide "
        "informative, concise, and friendly responses."
    )

    coding_prompt = (
        f"You are Chat99, an AI assistant powered by Claude {model_info['name'].title()} "
        "specializing in programming. You have extensive knowledge in software development. "
        "When asked coding questions, you provide clear, efficient, and well-commented solutions. "
        "You can work with multiple programming languages and explain complex concepts "
        "in an easy-to-understand manner. Format your code responses with markdown code blocks "
        "using triple backticks and the appropriate language identifier."
    )

    current_mode = "general"
    system_prompt = general_prompt

    conversation: List[Dict[str, str]] = []
    client = Anthropic()

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ").strip()

        if user_input.lower() == "exit":
            console.print(
                f"[bold green]{chat99_version}:[/bold green] Goodbye! "
                "It was nice chatting with you."
            )
            break

        if user_input.lower() == "switch":
            if current_mode == "general":
                current_mode = "coding"
                system_prompt = coding_prompt
                console.print("[bold yellow]Switched to coding mode.[/bold yellow]")
            else:
                current_mode = "general"
                system_prompt = general_prompt
                console.print(
                    "[bold yellow]Switched to general chat mode.[/bold yellow]"
                )
            continue

        conversation.append({"role": "user", "content": user_input})

        ai_response = generate_response(
            client,
            model_name,
            system_prompt,
            conversation,
            args.max_tokens,
            args.temperature,
        )

        if ai_response:
            console.print(f"[bold green]{chat99_version}:[/bold green] ", end="")
            console.print(ai_response)

            conversation.append({"role": "assistant", "content": ai_response})

            if len(conversation) > 10:
                conversation = conversation[-10:]

            display_message("assistant", ai_response)
        else:
            console.print(
                "[bold red]Failed to get a response. Please try again.[/bold red]"
            )


def check_api_key() -> bool:
    """Check if the Anthropic API key is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error: ANTHROPIC_API_KEY environment variable is not set.[/bold red]"
        )
        console.print("Please set your API key and try again.")
        return False
    return True


if __name__ == "__main__":
    if check_api_key():
        chat_args = parse_arguments()
        chat_with_99(chat_args)
