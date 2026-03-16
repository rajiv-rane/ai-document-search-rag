import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelevanceFilter:
    """
    Filters retrieved documents based on similarity scores and computes confidence.
    Operates independently of the LLM.
    """
    def __init__(self, threshold: float):
        """
        Initializes the filter with a relevance threshold.
        
        Args:
            threshold: Minimum required similarity score for the top result.
        """
        self.threshold = threshold

    def apply(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """
        Evaluates the relevance of retrieved chunks.
        Note: Current FAISS implementation uses L2 distance. 
        Higher similarity corresponds to lower L2 distance.
        The score is converted to a pseudo-similarity for comparison.
        
        Args:
            retrieved_chunks: List of dictionaries from the retriever.
            
        Returns:
            Tuple: (is_relevant: bool, confidence_score: float, message: str)
        """
        if not retrieved_chunks:
            return False, 0.0, "No documents found."

        # Convert L2 distances to pseudo-similarity scores (normalized 0 to 1 range)
        # s = 1 / (1 + distance) is a common heuristic
        similarities = [1.0 / (1.0 + chunk.get("score", 1.0)) for chunk in retrieved_chunks]
        
        highest_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)
        
        logger.info(f"Relevance Evaluation: Top Sim = {highest_similarity:.4f}, Threshold = {self.threshold}")

        if highest_similarity < self.threshold:
            return False, avg_similarity, "No relevant information found in provided documents."
            
        return True, avg_similarity, "Relevance check passed."
