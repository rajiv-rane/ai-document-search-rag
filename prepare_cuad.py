import os
import re
import logging
from pathlib import Path
from typing import List, Set

import datasets
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def clean_contract_text(text: str) -> str:
    """
    Cleans raw contract text by removing null characters, normalizing whitespace,
    and handling excessive newlines.

    Args:
        text: Raw text from the dataset.

    Returns:
        Cleaned text string.
    """
    # Remove null characters
    text = text.replace("\x00", "")
    
    # Normalize whitespace (replace multiple spaces/tabs with a single space)
    text = re.sub(r"[ \t]+", " ", text)
    
    # Remove excessive newlines (more than 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()

def prepare_cuad_data(
    output_dir: str = "dataset/cuad_contracts",
    max_contracts: int = 15,
    min_chars: int = 1000
):
    """
    Loads the CUAD dataset, extracts unique contracts, cleans them, 
    and saves them to the specified directory.

    Args:
        output_dir: Directory where cleaned .txt files will be saved.
        max_contracts: Number of unique contracts to save.
        min_chars: Minimum character length for a contract to be included.
    """
    logger.info("Initializing CUAD dataset preparation...")
    
    # Create directory if missing
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset (Hugging Face datasets handles caching)
        logger.info("Loading 'theatticusproject/cuad' dataset (train split)...")
        dataset = load_dataset("theatticusproject/cuad", split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    unique_contexts: Set[str] = set()
    saved_count = 0
    total_words = 0
    word_counts: List[int] = []

    logger.info("Processing contracts...")

    for item in dataset:
        if saved_count >= max_contracts:
            break
            
        context = item.get("context", "")
        
        # Check for uniqueness and minimum length
        if context and context not in unique_contexts and len(context) >= min_chars:
            unique_contexts.add(context)
            
            # Clean text
            cleaned_text = clean_contract_text(context)
            
            # Final length check after cleaning
            if len(cleaned_text) < min_chars:
                continue

            # Generate filename
            filename = f"CUAD_Contract_{saved_count + 1:03d}.txt"
            file_path = output_path / filename
            
            # Save file
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                
                # Calculate metrics
                words = cleaned_text.split()
                count = len(words)
                word_counts.append(count)
                total_words += count
                saved_count += 1
                
                logger.info(f"Saved: {filename} ({count} words)")
                
            except IOError as e:
                logger.error(f"Failed to save {filename}: {e}")

    # Results summary
    if saved_count > 0:
        avg_words = total_words / saved_count
        logger.info("--- Preparation Complete ---")
        logger.info(f"Total Unique Contracts Saved: {saved_count}")
        logger.info(f"Average Word Count: {avg_words:.2f}")
    else:
        logger.warning("No contracts were saved. Check dataset availability and filters.")

if __name__ == "__main__":
    prepare_cuad_data()
