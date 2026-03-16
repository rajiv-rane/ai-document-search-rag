import numpy as np
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    """
    High-level interface for document retrieval.
    Connects embedding logic with vector storage.
    """
    def __init__(self, vector_store: Any, embedder: Any, top_k: int = 3):
        """
        Initializes the retriever.
        
        Args:
            vector_store: An instance of VectorStore.
            embedder: An instance of DocumentEmbedder.
            top_k: Default number of chunks to retrieve.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Orchestrates the retrieval process: query embedding -> vector search.
        
        Args:
            query: The user's query string.
            top_k: Optional override for the number of results.
            
        Returns:
            List of retrieved chunks with metadata and similarity scores.
        """
        if not top_k:
            top_k = self.top_k
            
        logger.info(f"Retrieving top {top_k} results for query: '{query}'")
        
        # 1. Embed Query
        query_embedding = self.embedder.embed_text(query)
        
        # 2. Search Vector Store
        results = self.vector_store.search(query_embedding, k=top_k)
        
        return results
