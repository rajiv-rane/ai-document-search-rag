import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
PERSIST_DIRECTORY = BASE_DIR / "vector_store_data"

# Model Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "phi3:mini"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Retrieval Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3
RELEVANCE_THRESHOLD = 0.35  # Adjust based on observed similarity scores

# Metadata
PROJECT_NAME = "LocalDocIntel"
VERSION = "0.1.0"
