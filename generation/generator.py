import logging
from typing import List, Dict, Any
from .prompt_builder import PromptBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGGenerator:
    """
    Main orchestrator for the Generation phase of RAG.
    Constructs prompts and generates grounded answers using an LLM client.
    """
    def __init__(self, llm_client: Any):
        """
        Initializes the generator.
        
        Args:
            llm_client: An instance of OllamaClient (or any swappable LLM wrapper).
        """
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()

    def generate_grounded_answer(
        self, 
        query: str, 
        context_chunks: List[Dict[str, Any]], 
        confidence_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generates an answer based ONLY on the provided context retrieved from documents.
        
        Args:
            query: The user's question.
            context_chunks: List of retrieved text chunks with metadata.
            confidence_score: Calculation from the retrieval layer.
            
        Returns:
            Dict containing answer, source metadata, and confidence metrics.
        """
        # 1. Build the prompt
        system_prompt = self.prompt_builder.SYSTEM_INSTRUCTION
        user_prompt = self.prompt_builder.build_user_prompt(query, context_chunks)
        
        # 2. Get response from LLM
        answer = self.llm_client.generate_response(user_prompt, system_prompt=system_prompt)
        
        # 3. Extract source references (for documentation/attribution)
        sources = []
        for chunk in context_chunks:
            meta = chunk.get("metadata", {})
            sources.append({
                "file": meta.get("file_name"),
                "page": meta.get("page_number")
            })

        # Deduplicate sources
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]

        return {
            "answer": answer,
            "sources": unique_sources,
            "retrieval_confidence": round(confidence_score, 4)
        }
