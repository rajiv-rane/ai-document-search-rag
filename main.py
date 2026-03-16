import argparse
import logging
import sys
from pathlib import Path

# Load settings from config
from config import (
    DATASET_DIR, 
    PERSIST_DIRECTORY, 
    EMBEDDING_MODEL_NAME, 
    LLM_MODEL_NAME, 
    OLLAMA_BASE_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    RELEVANCE_THRESHOLD
)

# Import local modules
from ingestion.loader import DocumentLoader
from ingestion.chunker import TextChunker
from ingestion.embedder import DocumentEmbedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from guardrails.relevance_filter import RelevanceFilter
from generation.llm import OllamaClient
from generation.generator import RAGGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Local Document Intelligence System")
    parser.add_argument("--query", type=str, help="The question to ask about your documents")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vector index")
    args = parser.parse_args()

    # 1. Initialize Core Models/Services
    try:
        embedder = DocumentEmbedder(model_name=EMBEDDING_MODEL_NAME)
        vector_store = VectorStore(
            dimension=embedder.get_embedding_dimension(), 
            persist_directory=PERSIST_DIRECTORY
        )
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    
    # 2. Ingestion Logic
    # Rebuild if requested or if the index is currently empty
    if len(vector_store.metadata_map) == 0 or args.rebuild:
        logger.info("Starting document ingestion process...")
        
        loader = DocumentLoader(dataset_dir=DATASET_DIR)
        documents = loader.load_all()
        
        if not documents:
            logger.error(f"No documents found in {DATASET_DIR}. Please add .pdf or .txt files.")
            return

        chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = chunker.chunk_all(documents)
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_text(texts)
        
        vector_store.add_documents(embeddings, chunks)
        logger.info("Vector index successfully built and saved.")

    # 3. Query Execution
    if not args.query:
        logger.info("No query provided. Usage: python main.py --query 'your question here'")
        return

    # Retrieval
    retriever = Retriever(vector_store=vector_store, embedder=embedder, top_k=TOP_K)
    retrieved_chunks = retriever.retrieve(args.query)

    # Relevance Guardrail
    relevance_filter = RelevanceFilter(threshold=RELEVANCE_THRESHOLD)
    is_relevant, confidence, message = relevance_filter.apply(retrieved_chunks)

    if not is_relevant:
        print(f"\n[GUARDRAIL]: {message}")
        print(f"Retrieval Confidence: {confidence:.4f}")
        return

    # Generation
    llm_client = OllamaClient(model_name=LLM_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    generator = RAGGenerator(llm_client=llm_client)
    
    logger.info("Generating grounded answer...")
    result = generator.generate_grounded_answer(args.query, retrieved_chunks, confidence)

    # 4. Final Output Formatting
    print("\n" + "="*60)
    print(f"QUERY: {args.query}")
    print("="*60)
    print(f"\nANSWER:\n{result['answer']}")
    print("\nSOURCES:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source['file']} (Page {source['page']})")
    print(f"\nRETRIEVAL CONFIDENCE: {result['retrieval_confidence']}")
    print("="*60)

if __name__ == "__main__":
    main()
