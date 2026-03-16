import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Handles generation of text embeddings using SentenceTransformers.
    Optimized for CPU-only execution and batch processing.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initializes the embedding model.
        
        Args:
            model_name: The name of the SentenceTransformer model to use.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}...")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates embeddings for a single string or a list of strings.
        
        Args:
            text: Single string or list of strings to embed.
            
        Returns:
            Numpy array of embeddings.
        """
        if isinstance(text, str):
            text = [text]
            
        try:
            embeddings = self.model.encode(
                text, 
                show_progress_bar=len(text) > 10,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the generated embeddings.
        """
        return self.model.get_sentence_embedding_dimension()
