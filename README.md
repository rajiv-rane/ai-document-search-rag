# Ask Your Documents

A production-grade, CPU-only, local RAG (Retrieval-Augmented Generation) system for querying PDF and TXT documents. Built with **Ollama**, **FAISS**, and **Sentence-Transformers**.

This project provides both a CLI interface and a modern **Streamlit** web application to easily extract grounded answers from your own document collections.

## 🎯 Project Goals & Assessment Details

The goal of this project is to build an intelligent document parsing and Q&A system that:
- **Runs Fully Locally:** Ensuring data privacy by keeping all document intelligence on your own hardware (CPU-optimized).
- **Prevents Hallucinations:** Using strict prompting and retrieval-based guardrails to only answer based on the provided context. If the answer isn't in the text, it admits it doesn't know.
- **Provides Citation:** Every answer pinpoints the exact document and page number where the information was found.
- **Demonstrates Clean Architecture:** Built with modular components (Ingestion, Retrieval, Generation, Guardrails) for maintainability and scalability.

## 🏗️ Architecture

```text
[ Documents (.pdf/.txt) ] 
           |
           v
[ Ingestion (Loader) ] --> [ Chunker (800 tokens) ]
           |
           v
[ Sentence Transformers (all-MiniLM-L6-v2) ] --> [ Vector Store (FAISS) ]
           |                                             |
           +---------------------------------------------+
                               |
                               v
[ User Query ] --> [ Retriever ] --> [ Guardrails (Relevance Filter) ]
                                            |
                                            v (Success)
[ Ollama (phi3:mini) ] <---------------- [ Context Construction ]
           |
           v
[ Grounded Answer + Citations ]
```

## 🔄 Data Flow
1. **Ingestion**: Documents are recursively scanned and loaded from the `dataset/` directory. PDFs are parsed page-by-page via PyMuPDF.
2. **Chunking**: Text is split into 800-token chunks with a 150-token overlap to maintain context boundaries.
3. **Embedding**: Chunks are vectorized using a local `all-MiniLM-L6-v2` model.
4. **Storage**: Vectors and metadata are stored in a persisted FAISS index (`vector_store_data/`).
5. **Retrieval**: User queries are embedded and matched against the index using L2 distance (cosine similarity).
6. **Guardrails**: A relevance filter checks the similarity score; if below the configured threshold, it blocks the LLM to prevent hallucinations.
7. **Generation**: Relevant chunks are formatted into a rigid prompt for `phi3:mini` via the local Ollama API to construct the final answer.

## 🚀 Setup & Installation

### 1. Prerequisites (Ollama)
The system relies on a local LLM to generate the final answers. 
- Download and install [Ollama](https://ollama.com/).
- Open your terminal and pull the small-but-mighty language model:
   ```bash
   ollama pull phi3:mini
   ```

### 2. Clone & Install Dependencies
1. Clone this repository:
   ```bash
   git clone https://github.com/rajiv-rane/ask-your-docs.git
   cd ask-your-docs
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Add Your Documents
Create a folder named `dataset` in the root directory if it doesn't exist. Place any `.pdf` or `.txt` files inside it. The system will recursively scan all subfolders.

*(Optional)* You can run `python prepare_cuad.py` to attempt to download sample legal contracts for testing.

## 🖥️ Usage

### Streamlit UI (Recommended)
Run the interactive web interface, which features a configuration sidebar, progress indicators, and grounded chat:
```bash
streamlit run ui/app.py
```

### Command Line Interface (CLI)
You can directly query your documents from the terminal. The first run will automatically detect new files and build the vector index.
```bash
python main.py --query "What are the common termination triggers?"
```
*Use the `--rebuild` flag to force a re-index if you have changed the contents of your dataset directory.*

## 🛠️ Design Decisions

- **FAISS**: Chosen for its high performance and ease of local persistence. It is standard for CPU-optimized vector search.
- **all-MiniLM-L6-v2**: A highly efficient embedding model. It provides a perfect balance of speed on CPU and semantic accuracy for document retrieval.
- **phi3:mini**: Selected for its state-of-the-art performance in the "small model" category, making it ideal for local CPU-only RAG where reasoning is required but resources are limited.
- **Chunk Size (800)**: Carefully balanced to provide enough context for complex legal/technical queries while ensuring multiple chunks fit comfortably within the LLM's context window.

## ⚠️ Limitations
- **Scaling**: While efficient, extremely large datasets (10k+ pages) may encounter index build time overhead on low-power CPUs.
- **Local Dependencies**: Requires the Ollama service to be running locally in the background.

## 📈 Future Improvements
- **Hybrid Search**: Combining semantic search with BM25 keyword matching for better technical term retrieval.
- **Multi-Query Expansion**: Using the LLM to generate variations of user queries to overcome vocabulary mismatches.
- **Table Support**: Enhanced parsing for complex tables in structured PDFs.
