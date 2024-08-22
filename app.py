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
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
    render_template,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from advanced_router import advanced_router
from memory_manager import memory_manager
from search import perform_search
from rag import retrieve_relevant_info
from irac_framework import apply_irac_framework, apply_comparative_analysis
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

"""
app.py: Main module for the Monkey Chat AI assistant.

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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HIGH_TIER_MODEL = os.getenv("HIGH_TIER_MODEL", "claude-3-5-sonnet-20240620")
MID_TIER_MODEL = os.getenv("MID_TIER_MODEL", "llama-3.1-70b-versatile")
LOW_TIER_MODEL = os.getenv("LOW_TIER_MODEL", "llama-3.1-8b-instant")
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))

# Initialize API clients
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Initialize Groq client with error handling
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    logger.warning("GROQ_API_KEY is not set. Groq functionality will be limited.")
    groq_client = None

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "flowise"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
vector_db = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize LangChain Pinecone vectorstore
vectorstore = PineconeVectorStore(
    index=vector_db, embedding=embeddings, text_key="text"
)

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt", "pdf", "doc", "docx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        return jsonify({"message": f"File '{filename}' uploaded successfully"}), 200
    return jsonify({"error": "File type not allowed"}), 400


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    action = data.get("action", "chat")

    if not user_input:
        return jsonify({"error": "Missing message"}), 400

    try:
        if action == "chat":
            response = chat_with_99(user_input)
        elif action == "upsert-project":
            response = upsert_to_project(user_input)
        elif action == "upsert-all":
            response = upsert_to_all_chats(user_input)
        else:
            return jsonify({"error": "Invalid action"}), 400

        return jsonify({"response": response}), 200
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500


def upsert_to_project(user_input: str) -> str:
    # Implement logic to upsert the user input to the current project
    # For now, we'll just add it to the vectorstore
    vectorstore.add_texts([user_input])
    return f"Upserted to project: {user_input}"


def upsert_to_all_chats(user_input: str) -> str:
    # Implement logic to upsert the user input to all future chats
    # For now, we'll add it to both the vectorstore and the memory manager
    vectorstore.add_texts([user_input])
    memory_manager.add_to_long_term_memory(user_input)
    return f"Upserted to all chats: {user_input}"


@app.route("/health")
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
        # Get the latest user input
        user_input = conversation[-1]["content"]

        # Check if search is needed
        need_search = should_perform_search(user_input)

        # Get relevant context from memory and RAG
        context = memory_manager.get_relevant_context(user_input)
        rag_info = retrieve_relevant_info(user_input)
        search_results = perform_search(user_input) if need_search else ""

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
            if groq_client:
                api_response = groq_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                generated_response = api_response.choices[0].message.content
            else:
                logger.error("Groq client is not initialized. Cannot use Groq models.")
                return "I'm sorry, but I'm currently unable to use Groq models. Please try again later or use a different model."
        else:
            logger.error(f"Invalid model specified: {model}")
            return "I'm sorry, but I encountered an error while processing your request. Please try again."

        return generated_response

    except Exception as e:
        logger.error(
            f"Unexpected error in generate_response: {str(e)}\n{traceback.format_exc()}"
        )
        return "An unexpected error occurred. Please try again or contact support if the issue persists."


def should_perform_search(user_input: str) -> bool:
    """
    Determine if a search should be performed based on the user input.
    """
    # List of common greetings and short phrases that don't require a search
    no_search_phrases = ["hello", "hi", "hey", "greetings", "how are you", "what's up"]

    # Convert input to lowercase for case-insensitive comparison
    lower_input = user_input.lower()

    # Check if the input is a greeting or short phrase
    if any(phrase in lower_input for phrase in no_search_phrases):
        return False

    # Check if the input is too short (e.g., less than 3 words)
    if len(user_input.split()) < 3:
        return False

    # If none of the above conditions are met, perform a search
    return True


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


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        memory_manager.cleanup_memory()
        if hasattr(github_manager, "cleanup"):
            github_manager.cleanup()
        if hasattr(github_manager, "close"):
            github_manager.close()
