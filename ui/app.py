import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

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

from ingestion.loader import DocumentLoader
from ingestion.chunker import TextChunker
from ingestion.embedder import DocumentEmbedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from guardrails.relevance_filter import RelevanceFilter
from generation.llm import OllamaClient
from generation.generator import RAGGenerator

# Page config
st.set_page_config(page_title="Local Doc Intel", layout="wide")

st.title("📄 Local Document Intelligence System")
st.markdown("---")

# Sidebar for configuration and ingestion
with st.sidebar:
    st.header("Configuration")
    dataset_path = st.text_input("Dataset Directory", value=str(DATASET_DIR))
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
    llm_client = OllamaClient(model_name=LLM_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    generator = RAGGenerator(llm_client=llm_client)
    relevance_filter = RelevanceFilter(threshold=RELEVANCE_THRESHOLD)
    return embedder, vector_store, generator, relevance_filter

embedder, vector_store, generator, relevance_filter = get_backend()

# Indexing Logic
if rebuild_index or len(vector_store.metadata_map) == 0:
    with st.status("Building Vector Index...", expanded=True) as status:
        st.write("Loading documents...")
        loader = DocumentLoader(dataset_dir=Path(dataset_path))
        documents = loader.load_all()
        
        if not documents:
            st.error("No documents found! Please check the directory.")
        else:
            st.write(f"Chunking {len(documents)} document segments...")
            chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = chunker.chunk_all(documents)
            
            st.write(f"Generating embeddings for {len(chunks)} chunks...")
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_text(texts)
            
            vector_store.add_documents(embeddings, chunks)
            st.success("Index ready!")
            status.update(label="Index built successfully!", state="complete")

# Query UI
query = st.text_input("Ask a question about your documents:", placeholder="e.g., What is the notice period for termination?")

if query:
    with st.spinner("Retrieving and generating answer..."):
        # Retrieval
        retriever = Retriever(vector_store=vector_store, embedder=embedder, top_k=TOP_K)
        retrieved_chunks = retriever.retrieve(query)
        
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
