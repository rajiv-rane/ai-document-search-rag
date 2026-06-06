import logging
from typing import List, Dict, Any, Generator
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Wrapper for interacting with OpenAI-compatible APIs (OpenAI, Groq, Ollama).
    """
    def __init__(self, model_name: str, base_url: str, api_key: str = None):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "sk-dummy"
        )

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generates a text completion from the specified model.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.info(f"Sending request to LLM ({self.model_name})...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                stream=False
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return f"Error: Failed to connect to LLM ({self.model_name}). Details: {str(e)}"

    def stream_response(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        """
        Streams completions from the model bit by bit.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                        
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield "Streaming failed."
