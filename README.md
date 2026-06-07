---
title: AI Docs Search RAG
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI-Powered Document Search & Chat (RAG)

A production-grade, local Retrieval-Augmented Generation (RAG) system for querying PDF and TXT documents. Built with Ollama, FAISS, and Sentence-Transformers.

This project provides a modern Streamlit web application to easily extract grounded answers from your document collections, featuring hybrid cloud/local deployment options, strict hallucination guardrails, and real-time ingestion deduplication.

## Project Goals & Architecture

The primary objective of this project is to build an intelligent document parsing and Q&A system that:
- **Ensures Data Privacy:** Capable of running fully locally on CPU-optimized hardware.
- **Prevents Hallucinations:** Utilizes strict prompting, rigid relevance thresholds, and retrieval-based guardrails to answer based exclusively on the provided context.
- **Provides Explicit Citations:** Every answer pinpoints the exact document filename and page number where the information was sourced.
- **Demonstrates Clean Architecture:** Built with modular components (Ingestion, Retrieval, Generation, Guardrails) for maintainability and scalability.

### System Architecture Diagram

![System Architecture Diagram](architecture_diagram.png)

## Data Flow & Features

1. **Ingestion & Deduplication**: Users upload documents via the Streamlit UI. The system runs a structural deduplication check against the dataset directory to prevent duplicate indexing and wasted compute. 
2. **Chunking**: Raw text is parsed (using PyMuPDF for PDFs) and split into 800-token chunks with a 150-token overlap to maintain context boundaries.
3. **Embedding**: Chunks are vectorized using the highly efficient `all-MiniLM-L6-v2` model.
4. **Storage**: Vectors and metadata are stored persistently in a local FAISS index.
5. **Retrieval**: User queries are embedded and matched against the FAISS index using L2 distance scoring.
6. **Validation & Guardrails**: The system applies a hard cutoff similarity threshold (L2 distance > 1.25). If retrieved chunks are completely irrelevant, the LLM is bypassed entirely to save resources. A secondary relevance filter computes confidence scores.
7. **Generation**: Relevant chunks are formatted into a rigid prompt. The system uses an OpenAI-compatible client, supporting both local execution (Ollama with `phi3:mini`) and cloud execution (Groq with `llama-3.3-70b-versatile`).

## Setup & Installation

### 1. Prerequisites
You can run the system locally using Ollama or via cloud using Groq.
- For local execution, install [Ollama](https://ollama.com/) and pull the required model:
   ```bash
   ollama pull phi3:mini
   ```

### 2. Clone & Install Dependencies
1. Clone this repository and navigate to the project root:
   ```bash
   git clone https://github.com/rajiv-rane/ask-your-docs.git
   cd ask-your-docs
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configuration (Optional)
The system defaults to Local Mode. To use Cloud Mode for faster generation, set the following environment variables before starting the application:
```powershell
# Windows
$env:DEPLOYMENT_MODE="cloud"
$env:GROQ_API_KEY="your-groq-api-key"
```

## Usage

Start the interactive web application:
```bash
streamlit run ui/app.py
```

### Working with the Dataset
Upload your own `.pdf` or `.txt` files directly through the Streamlit sidebar. The system will automatically chunk, embed, and index them in real-time.

A comprehensive sample dataset containing CUAD contracts, SEC filings, and various legal agreements is available here:
[Dataset Google Drive Link](https://drive.google.com/drive/folders/1V_JAWu9wEn1aWxrgFi3wXdyKDFNDCYlF?usp=sharing)

You can download these files and upload them via the web interface to test the system's capabilities.

## Design Decisions

- **FAISS**: Chosen for high performance and ease of local persistence. It is the standard for CPU-optimized vector search.
- **all-MiniLM-L6-v2**: Provides a perfect balance of speed on CPU and semantic accuracy for document retrieval.
- **phi3:mini / Llama-3.3-70b**: Phi-3 provides state-of-the-art performance for local CPU-only environments, while Llama 3.3 via Groq offers lightning-fast cloud generation.
- **Chunk Size (800)**: Carefully balanced to provide enough context for complex legal queries while fitting within context windows.

## Limitations

- **Scaling**: Extremely large datasets (10k+ pages) may encounter index build time overhead on low-power CPUs during the vectorization phase.
- **Local Dependencies**: Local mode requires the Ollama service running in the background.

## Future Improvements

- **Hybrid Search**: Combining semantic search with BM25 keyword matching for better technical term retrieval.
- **Multi-Query Expansion**: Using the LLM to generate variations of user queries to overcome vocabulary mismatches.
- **Table Support**: Enhanced parsing for complex tables in structured PDFs.
