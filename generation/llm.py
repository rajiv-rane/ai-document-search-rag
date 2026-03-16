import requests
import json
import logging
from typing import List, Dict, Any, Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Wrapper for interacting with the local Ollama LLM service.
    """
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{self.base_url}/api/generate"

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generates a text completion from the specified model.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0  # Keep it deterministic for RAG
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        try:
            logger.info(f"Sending request to Ollama ({self.model_name})...")
            response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=60,
                proxies={"http": None, "https": None}
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: Failed to connect to local LLM ({self.model_name}). Ensure Ollama is running."

    def stream_response(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        """
        Streams completions from the model bit by bit.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.0}
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                self.api_url, 
                json=payload, 
                stream=True,
                proxies={"http": None, "https": None}
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if not chunk.get("done", False):
                        yield chunk.get("response", "")
                        
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield "Streaming failed."
