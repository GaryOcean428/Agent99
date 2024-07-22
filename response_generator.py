from anthropic import Anthropic
from input_analyzer import analyze_input

client = Anthropic()

def generate_response(model_name, system_prompt, conversation, max_tokens, temperature):
    user_input = conversation[-1]['content']
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
        # Ensure the conversation history alternates between user and assistant
        cleaned_conversation = []
        for i, message in enumerate(conversation):
            if i % 2 == 0 and message['role'] != 'user':
                continue
            if i % 2 == 1 and message['role'] != 'assistant':
                continue
            cleaned_conversation.append(message)

        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=cleaned_conversation
        )
        return response
    except Exception as e:
        print(f"An error occurred in bot response: {str(e)}")
        return None