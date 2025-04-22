"""
LLM Service - Handles interactions with language models via Together API
"""
import together
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Together client
api_key = os.getenv("TOGETHER_API_KEY")
if api_key:
    together.api_key = api_key


def prompt_llm(prompt, show_cost=True):
    """
    Function to send prompt to an LLM via the Together API.
    
    Args:
        prompt (str): The prompt text to send to the LLM
        show_cost (bool): Whether to display token count and cost estimate
        
    Returns:
        str: The LLM's response text
    """
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
    tokens = len(prompt.split())

    if show_cost:
        print(f"\nNumber of tokens: {tokens}")
        cost = (0.1 / 1_000_000) * tokens
        print(f"Estimated cost for {model}: ${cost:.10f}\n")

    try:
        response = together.Complete.create(
            prompt=prompt,
            model=model,
            max_tokens=512,
            temperature=0.7,
            top_k=50,
            top_p=0.7,
            repetition_penalty=1.1,
        )

        content = response['output']['choices'][0]['text']

        if not content or len(content.strip()) < 10:
            print(f"Warning: LLM returned empty or very short response: '{content}'")
            return "The LLM response was too short or empty. Please try again with more detailed input."
        return content.strip()
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        return "An error occurred while generating content. Please check your API key and try again." 