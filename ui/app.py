import streamlit as st
import os
import sys
import logging
from pathlib import Path
import importlib

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
importlib.reload(config)

from config import (
    DATASET_DIR, 
    PERSIST_DIRECTORY, 
    EMBEDDING_MODEL_NAME, 
    LLM_MODEL_NAME, 
    OLLAMA_BASE_URL,
    GROQ_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    RELEVANCE_THRESHOLD,
    SIMILARITY_THRESHOLD
)

from ingestion.loader import DocumentLoader, PDFLoader, TextLoader
from ingestion.chunker import TextChunker
from ingestion.embedder import DocumentEmbedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from guardrails.relevance_filter import RelevanceFilter
from generation.llm import OpenAIClient
from generation.generator import RAGGenerator

# Page config
st.set_page_config(page_title="AI Doc Search", layout="wide")

st.title("📄 AI-Powered Document Search & Chat (RAG)")
st.markdown("---")

# Sidebar for configuration and ingestion
with st.sidebar:
    st.header("Configuration")
    dataset_path = st.text_input("Dataset Directory", value=str(DATASET_DIR))
    
    uploaded_files = st.file_uploader("Upload Documents (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    
    rebuild_index = st.button("🔥 Rebuild Vector Index")
    
    st.info(f"Model: {LLM_MODEL_NAME}\nEmbeddings: {EMBEDDING_MODEL_NAME}")

# Initialize backend components
@st.cache_resource
def get_backend():
    embedder = DocumentEmbedder(model_name=EMBEDDING_MODEL_NAME)
    vector_store = VectorStore(
        dimension=embedder.get_embedding_dimension(), 
        persist_directory=PERSIST_DIRECTORY
    )
    llm_client = OpenAIClient(model_name=LLM_MODEL_NAME, base_url=OLLAMA_BASE_URL, api_key=GROQ_API_KEY)
    generator = RAGGenerator(llm_client=llm_client)
    relevance_filter = RelevanceFilter(threshold=RELEVANCE_THRESHOLD)
    return embedder, vector_store, generator, relevance_filter

embedder, vector_store, generator, relevance_filter = get_backend()

# 1. Structural File Deduplication and Ingestion
if uploaded_files:
    os.makedirs(dataset_path, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(dataset_path, uploaded_file.name)
        
        # Deduplication Check
        if os.path.exists(file_path):
            st.sidebar.warning(f"ℹ️ {uploaded_file.name} is already fully indexed and ready for querying!")
        else:
            # Write new file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Real-time Interactive Vectorization
            with st.spinner(f"Vectorizing new document: {uploaded_file.name}..."):
                suffix = Path(file_path).suffix.lower()
                docs = []
                if suffix == ".pdf":
                    docs = PDFLoader().load(Path(file_path))
                elif suffix == ".txt":
                    docs = TextLoader().load(Path(file_path))
                    
                if docs:
                    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    chunks = chunker.chunk_all(docs)
                    texts = [c["text"] for c in chunks]
                    embeddings = embedder.embed_text(texts)
                    vector_store.add_documents(embeddings, chunks)
                    st.sidebar.success(f"✅ Success: {uploaded_file.name} indexed!")

# Indexing Logic for full rebuild or first initialization
if rebuild_index or len(vector_store.metadata_map) == 0:
    msg = "Rebuilding Vector Index..." if rebuild_index else "Initializing Vector Index..."
    with st.spinner(msg):
        loader = DocumentLoader(dataset_dir=Path(dataset_path))
        documents = loader.load_all()
        
        if not documents:
            st.error("No documents found! Please check the directory or upload files.")
        else:
            chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = chunker.chunk_all(documents)
            
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_text(texts)
            
            vector_store.add_documents(embeddings, chunks)
            st.sidebar.success("Index ready!")

# Query UI
query = st.text_input("Ask a question about your documents:", placeholder="e.g., What is the notice period for termination?")

if query:
    with st.spinner("Retrieving and generating answer..."):
        # Retrieval
        retriever = Retriever(vector_store=vector_store, embedder=embedder, top_k=TOP_K)
        retrieved_chunks = retriever.retrieve(query)
        
        # Validation check against SIMILARITY_THRESHOLD
        if retrieved_chunks:
            best_score = min([chunk.get('score', float('inf')) for chunk in retrieved_chunks])
            logging.info(f"Retrieval Validation: Best L2 Distance = {best_score:.4f}, Threshold = {SIMILARITY_THRESHOLD}")
            
            if best_score > SIMILARITY_THRESHOLD:
                logging.warning(f"Bypassing LLM: Best score ({best_score:.4f}) indicates chunks are completely irrelevant.")
                st.info("I'm sorry, no relevant context could be retrieved from the dataset to answer your query.")
            else:
                # Guardrail
                is_relevant, confidence, message = relevance_filter.apply(retrieved_chunks)
                
                if not is_relevant:
                    st.warning(f"**Guardrail Alert:** {message}")
                    st.write(f"Confidence score ({confidence:.4f}) was below threshold.")
                else:
                    # Generation
                    result = generator.generate_grounded_answer(query, retrieved_chunks, confidence)
                    
                    # Display Result
                    st.subheader("Answer")
                    st.write(result['answer'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Sources")
                        for src in result['sources']:
                            st.markdown(f"- **{src['file']}** (Page {src['page']})")
                    
                    with col2:
                        st.subheader("Metrics")
                        st.metric("Retrieval Confidence", f"{result['retrieval_confidence']:.4f}")

st.markdown("---")
st.caption("Powered by Ollama (phi3:mini) and FAISS")
