"""
This module handles the generation of responses for the Chat99 AI assistant.
It integrates with the Anthropic API and manages the conversation flow.
"""

from typing import List, Dict, Any
from anthropic import Anthropic
from input_analyzer import analyze_input

client = Anthropic()

def generate_response(
    model_name: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float
) -> str:
    """
    Generate a response from the AI model using the Anthropic API.

    Args:
        model_name (str): The name of the AI model to use.
        system_prompt (str): The system prompt to guide the AI's behavior.
        messages (List[Dict[str, str]]): The conversation history.
        max_tokens (int): The maximum number of tokens in the response.
        temperature (float): The randomness of the response.

    Returns:
        str: The generated response from the AI model.
    """
    user_input = messages[-1]['content']
    input_type = analyze_input(user_input)
    
    if input_type == "complex":
        thought_process = """
        1. Understand the question or task
        2. Break down the problem into smaller parts
        3. Consider relevant information and context
        4. Analyze potential approaches or solutions
        5. Draw conclusions or provide a step-by-step explanation
        6. Summarize the response
        """
        
        system_prompt += f"\n\nFor complex queries, follow this thought process:\n{thought_process}"
    
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text
    except (Anthropic.APIError, Anthropic.APIConnectionError) as e:
        print(f"An API error occurred: {str(e)}")
    except Anthropic.AuthenticationError:
        print("Authentication error: Please check your API key.")
    except Anthropic.RateLimitError:
        print("Rate limit exceeded: Please try again later.")
    except Anthropic.APIStatusError as e:
        print(f"API status error: {str(e)}")
    except KeyError:
        print("Unexpected response format from the API.")
    except ValueError:
        print("Invalid value in the API response.")
    return ""