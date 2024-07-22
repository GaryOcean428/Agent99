import os
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from models import get_model_info, get_model_list
from response_generator import generate_response

# Set up Rich console
console = Console()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Chat99 (powered by Claude)")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens for response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation")
    return parser.parse_args()

def display_message(role, content):
    if role == "user":
        console.print(Panel(content, expand=False, border_style="blue", title="You"))
    else:
        md = Markdown(content)
        console.print(Panel(md, expand=False, border_style="green", title="Chat99"))

def chat_with_99(args):
    console.print(Panel("Welcome to Chat99 (powered by Claude)!", title="Chat Interface", border_style="bold magenta"))
    
    # Ask for preferred model at startup
    console.print("Available models:")
    for key, model in get_model_list().items():
        console.print(f"{key}. {model['name']}")
    
    preferred_model = Prompt.ask("Enter the number of your preferred model", choices=list(get_model_list().keys()))
    model_info = get_model_info(preferred_model)

    model_name = model_info['id']
    chat99_version = f"Chat99 {model_info['name'].title()}"
    console.print(f"[bold green]Using {chat99_version}[/bold green]")
    console.print("Type 'exit' to end the conversation, 'switch' to change chat mode.")
    
    general_prompt = f"""You are Chat99, an AI assistant powered by Claude {model_info['name'].title()}. 
    You are helpful, honest, and harmless. You have extensive knowledge in various fields and can engage in 
    conversations on a wide range of topics. You provide informative, concise, and friendly responses."""

    coding_prompt = f"""You are Chat99, an AI assistant powered by Claude {model_info['name'].title()} specializing in programming. 
    You have extensive knowledge in software development. When asked coding questions, you provide clear, efficient, 
    and well-commented solutions. You can work with multiple programming languages and explain complex concepts 
    in an easy-to-understand manner. Format your code responses with markdown code blocks using triple backticks 
    and the appropriate language identifier."""

    current_mode = "general"
    system_prompt = general_prompt
    
    conversation = []
  
    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        
        if user_input.lower() == 'exit':
            console.print(f"[bold green]{chat99_version}:[/bold green] Goodbye! It was nice chatting with you.")
            break
        
        if user_input.lower() == 'switch':
            if current_mode == "general":
                current_mode = "coding"
                system_prompt = coding_prompt
                console.print("[bold yellow]Switched to coding mode.[/bold yellow]")
            else:
                current_mode = "general"
                system_prompt = general_prompt
                console.print("[bold yellow]Switched to general chat mode.[/bold yellow]")
            continue
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})
        
        try:
            response = generate_response(model_name, system_prompt, conversation, args.max_tokens, args.temperature)
            
            if response and response.content:
                ai_response = response.content[0].text
                
                console.print(f"[bold green]{chat99_version}:[/bold green] ", end="")
                console.print(ai_response)
                
                # Add assistant message to conversation
                conversation.append({"role": "assistant", "content": ai_response})
                
                # Trim conversation history if it gets too long
                if len(conversation) > 10:
                    conversation = conversation[-10:]
                
                # Display formatted response
                display_message("assistant", ai_response)
                
                # Print token usage
                console.print(Panel(
                    f"Input Tokens: {response.usage.input_tokens}\n"
                    f"Output Tokens: {response.usage.output_tokens}\n"
                    f"Total Tokens: {response.usage.input_tokens + response.usage.output_tokens}",
                    title="Token Usage", border_style="cyan", expand=False
                ))
            else:
                console.print("[bold red]Failed to get a response. Please try again.[/bold red]")
        
        except Exception as e:
            console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
            console.print("Please try again.")

def check_api_key():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[bold red]Error: ANTHROPIC_API_KEY environment variable is not set.[/bold red]")
        console.print("Please set your API key and try again.")
        return False
    return True

if __name__ == "__main__":
    if check_api_key():
        args = parse_arguments()
        chat_with_99(args)