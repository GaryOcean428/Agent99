import os
import logging
import traceback
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from anthropic import Anthropic
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import requests
from flask import Flask, jsonify
from advanced_router import advanced_router
from memory_manager import memory_manager
from search import perform_search
from rag import retrieve_relevant_info
from irac_framework import apply_irac_framework, apply_comparative_analysis

"""
chat99.py: Main module for the Chat99 AI assistant.

This module integrates various components including advanced routing,
memory management, retrieval-augmented generation (RAG), and web search
to provide a sophisticated chatbot experience. It supports multiple
language models and adaptive response strategies.
"""


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up Rich console
console = Console()

# Load configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HIGH_TIER_MODEL = os.getenv("HIGH_TIER_MODEL", "claude-3-5-sonnet-20240620")
MID_TIER_MODEL = os.getenv("MID_TIER_MODEL", "llama-3.1-70b-versatile")
LOW_TIER_MODEL = os.getenv("LOW_TIER_MODEL", "llama-3.1-8b-instant")
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))

# Initialize API clients
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Flask app
app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

def generate_response(
    model: str,
    conversation: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    response_strategy: str = "default",
) -> str:
    try:
        # Get relevant context from memory and RAG
        context = memory_manager.get_relevant_context(conversation[-1]["content"])
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
            api_response = anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            generated_response = api_response.content[0].text
        elif model in [MID_TIER_MODEL, LOW_TIER_MODEL]:
            api_response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            generated_response = api_response.choices[0].message.content
        else:
            logger.error(f"Invalid model specified: {model}")
            return "I'm sorry, but I encountered an error while processing your request. Please try again."

        return generated_response

    except Exception as e:
        logger.error(
            f"Unexpected error in generate_response: {str(e)}\n{traceback.format_exc()}"
        )
        return "An unexpected error occurred. Please try again or contact support if the issue persists."


def get_strategy_instruction(strategy: str) -> str:
    """
    Get the instruction for the specified response strategy.
    """
    strategies = {
        "casual_conversation": "Respond in a casual, friendly manner without using any specific format.",
        "chain_of_thought": "Use a step-by-step reasoning approach. Break down the problem, consider relevant information, and explain your thought process clearly.",
        "direct_answer": "Provide a concise, direct answer to the question without unnecessary elaboration.",
        "boolean_with_explanation": "Start with a clear Yes or No, then provide a brief explanation for your answer.",
        "open_discussion": "Engage in an open-ended discussion, providing insights and asking follow-up questions.",
        "comparative_analysis": "Provide a detailed comparison and analysis of the given topics or concepts.",
    }
    return strategies.get(
        strategy,
        "Respond naturally to the query, providing relevant information and insights.",
    )


class GitHubManager:
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def list_repositories(self) -> List[Dict[str, Any]]:
        """List repositories for the authenticated user."""
        response = requests.get(
            f"{self.base_url}/user/repos", headers=self.headers, timeout=5
        )
        response.raise_for_status()
        return response.json()

    def create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, body: str
    ) -> Dict[str, Any]:
        """Create a pull request."""
        data = {"title": title, "head": head, "base": base, "body": body}
        response = requests.post(
            f"{self.base_url}/repos/{owner}/{repo}/pulls",
            headers=self.headers,
            json=data,
            timeout=5,
        )
        response.raise_for_status()
        return response.json()

    def merge_pull_request(
        self, owner: str, repo: str, pull_number: int
    ) -> Dict[str, Any]:
        """Merge a pull request."""
        response = requests.put(
            f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}/merge",
            headers=self.headers,
            timeout=5,
        )
        response.raise_for_status()
        return response.json()

    def cleanup(self):
        """Cleanup method for GitHubManager."""
        logger.info("GitHubManager cleanup completed")

    def close(self):
        """Close method for GitHubManager."""
        logger.info("GitHubManager closed")


github_manager = GitHubManager()


def handle_github_commands(user_input: str) -> str:
    """Handle GitHub-related commands."""
    if user_input.startswith("list repos"):
        repos = github_manager.list_repositories()
        return "Your repositories:\n" + "\n".join([repo["full_name"] for repo in repos])

    elif user_input.startswith("create pr"):
        # Example: "create pr owner/repo title head base body"
        parts = user_input.split(maxsplit=6)
        if len(parts) != 7:
            return "Invalid command. Use: create pr owner/repo title head base body"
        _, _, repo, title, head, base, body = parts
        owner, repo = repo.split("/")
        pr = github_manager.create_pull_request(owner, repo, title, head, base, body)
        return f"Pull request created: {pr['html_url']}"

    elif user_input.startswith("merge pr"):
        # Example: "merge pr owner/repo pull_number"
        parts = user_input.split()
        if len(parts) != 4:
            return "Invalid command. Use: merge pr owner/repo pull_number"
        _, _, repo, pull_number = parts
        owner, repo = repo.split("/")
        result = github_manager.merge_pull_request(owner, repo, int(pull_number))
        return f"Pull request merged: {result['message']}"

    return "Unknown GitHub command"


def chat_with_99(
    user_input: str, conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    if conversation_history is None:
        conversation_history = []

    conversation_history.append({"role": "user", "content": user_input})

    try:
        # Check for GitHub commands
        if user_input.lower().startswith(("list repos", "create pr", "merge pr")):
            return handle_github_commands(user_input)

        # Check for similar queries in memory
        similar_query = memory_manager.get_relevant_context(user_input)
        if similar_query:
            logger.info("Found similar query in memory")
            return similar_query

        route_config = advanced_router.route(user_input, conversation_history)
        model = route_config.get("model", HIGH_TIER_MODEL)
        max_tokens = route_config.get("max_tokens", 1024)
        temperature = route_config.get("temperature", 0.7)
        response_strategy = route_config.get("response_strategy", "default")

        logger.info("Routing decision: %s", route_config)
        console.print(
            Panel(
                f"[bold cyan]Model: {model}[/bold cyan]\n[bold yellow]Strategy: {response_strategy}[/bold yellow]"
            )
        )

        generated_response = generate_response(
            model, conversation_history, max_tokens, temperature, response_strategy
        )

        conversation_history.append(
            {"role": "assistant", "content": generated_response}
        )

        return generated_response
    except Exception as e:
        logger.error(
            "An error occurred in chat_with_99: %s\n%s", str(e), traceback.format_exc()
        )
        return "I'm sorry, but an error occurred while processing your request. Please try again later."


def display_chat_summary(chat_history: List[Dict[str, str]]) -> None:
    """
    Display a summary of the chat history.
    """
    console.print("\n[bold]Chat Summary:[/bold]")
    for entry in chat_history:
        role = entry["role"]
        content = (
            entry["content"][:50] + "..."
            if len(entry["content"]) > 50
            else entry["content"]
        )
        if role == "user":
            console.print(f"[blue]User:[/blue] {content}")
        else:
            console.print(f"[green]Assistant:[/green] {content}")


def main() -> None:
    console.print(
        "[bold]Welcome to Chat99! Type 'exit' to end the conversation, or 'summary' to see chat history.[/bold]"
    )
    chat_history: List[Dict[str, str]] = []

    while True:
        user_message = console.input("[bold blue]You:[/bold blue] ")

        if user_message.lower() == "exit":
            console.print(
                "[bold green]Chat99:[/bold green] Goodbye! It was nice chatting with you."
            )
            break
        if user_message.lower() == "summary":
            display_chat_summary(chat_history)
            continue

        try:
            response = chat_with_99(user_message, chat_history)
            console.print(
                Panel(
                    Markdown(f"[bold green]Chat99:[/bold green] {response}"),
                    expand=False,
                )
            )

            # Update chat history
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response})

            # Trim conversation history if it exceeds the limit
            if len(chat_history) > MAX_CONVERSATION_HISTORY:
                chat_history = chat_history[-MAX_CONVERSATION_HISTORY:]
        except Exception as e:
            logger.error(
                "An unexpected error occurred: %s\n%s", str(e), traceback.format_exc()
            )
            console.print(
                "[bold red]An unexpected error occurred. Please try again.[/bold red]"
            )


if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        memory_manager.cleanup_memory()
        if hasattr(github_manager, "cleanup"):
            github_manager.cleanup()
        if hasattr(github_manager, "close"):
            github_manager.close()
