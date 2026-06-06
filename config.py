import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
PERSIST_DIRECTORY = BASE_DIR / "vector_store_data"

# Deployment Mode
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local")

# Model Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

if DEPLOYMENT_MODE == "cloud":
    LLM_MODEL_NAME = "llama-3.3-70b-versatile"
    OLLAMA_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
else:
    LLM_MODEL_NAME = "phi3:mini"
    OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
    GROQ_API_KEY = None

# Retrieval Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3
RELEVANCE_THRESHOLD = 0.35  # Adjust based on observed similarity scores
SIMILARITY_THRESHOLD = 1.25

# Metadata
PROJECT_NAME = "AIDocSearchRAG"
VERSION = "0.1.0"
