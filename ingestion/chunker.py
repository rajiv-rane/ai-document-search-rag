import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    """
    Handles splitting of document text into overlapping chunks.
    Designed to preserve metadata and handle large volumes of text efficiently.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initializes the chunker with specified size and overlap.
        
        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of overlapping tokens between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks a single document entry (e.g., a page).
        
        Args:
            doc_data: Dictionary containing 'text', 'doc_id', 'file_name', 'page_number'.
            
        Returns:
            List of chunk dictionaries with updated metadata and chunk_id.
        """
        text = doc_data.get("text", "")
        if not text:
            return []

        # Simple word-based tokenization as requested
        tokens = text.split()
        num_tokens = len(tokens)
        chunks = []
        
        if num_tokens == 0:
            return []

        # Sliding window approach
        start = 0
        chunk_idx = 0
        
        while start < num_tokens:
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            
            # Create chunk object
            chunk_id = f"{doc_data['doc_id']}_p{doc_data['page_number']}_c{chunk_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_data["doc_id"],
                    "file_name": doc_data["file_name"],
                    "page_number": doc_data["page_number"],
                    "chunk_index": chunk_idx,
                    "token_count": len(chunk_tokens)
                }
            })
            
            # Move window: if end is past total tokens, we're done
            if end >= num_tokens:
                break
                
            start += (self.chunk_size - self.chunk_overlap)
            chunk_idx += 1
            
        return chunks

    def chunk_all(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of documents and returns a flattened list of all chunks.
        """
        all_chunks = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            
        logger.info(f"Generated {len(all_chunks)} chunks from {len(documents)} document entries.")
        return all_chunks
