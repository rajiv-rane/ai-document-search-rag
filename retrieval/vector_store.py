import os
import faiss
import numpy as np
import pickle
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages a FAISS index and associated metadata.
    Supports persistent storage and loading from disk.
    """
    def __init__(self, dimension: int, persist_directory: Path):
        """
        Initializes the vector store.
        
        Args:
            dimension: Dimension of the vectors to be stored.
            persist_directory: Path to save/load the index and metadata.
        """
        self.dimension = dimension
        self.persist_directory = persist_directory
        self.index_path = persist_directory / "faiss.index"
        self.metadata_path = persist_directory / "metadata.pkl"
        
        # Initialize index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_map: List[Dict[str, Any]] = []

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Auto-load if existing
        if self.index_path.exists():
            self.load_index()

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """
        Adds document chunks and their embeddings to the store.
        
        Args:
            embeddings: Numpy array of shape (num_docs, dimension).
            documents: List of chunk dictionaries containing text and metadata.
        """
        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of embeddings must match number of documents.")
            
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}.")

        # Add to FAISS
        self.index.add(embeddings.astype("float32"))
        
        # Add to metadata map
        self.metadata_map.extend(documents)
        logger.info(f"Added {len(documents)} documents to the index.")
        
        # Save after adding
        self.save_index()

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs similarity search against the index.
        
        Args:
            query_embedding: Numpy array of shape (1, dimension).
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of dictionaries containing matched text, score, and metadata.
        """
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[1]} does not match index dimension {self.dimension}.")

        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata_map):
                continue
                
            entry = self.metadata_map[idx].copy()
            entry["score"] = float(dist)  # L2 distance (lower is better)
            results.append(entry)
            
        return results

    def save_index(self):
        """Saves index and metadata map to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata_map, f)
            logger.info(f"Vector store saved to {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load_index(self):
        """Loads index and metadata map from disk."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                self.metadata_map = pickle.load(f)
            logger.info(f"Loaded existing vector store from {self.persist_directory} with {len(self.metadata_map)} items.")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            # Reset to clean state if load fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata_map = []
