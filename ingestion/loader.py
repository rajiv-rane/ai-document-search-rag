import logging
import fitz  # PyMuPDF
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Orchestrates the loading of documents from a directory.
    """
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.loaders = {
            ".pdf": PDFLoader(),
            ".txt": TextLoader()
        }

    def load_all(self) -> List[Dict[str, Any]]:
        """
        Recursively scans the dataset directory and loads all supported files.
        """
        all_documents = []
        
        if not self.dataset_dir.exists():
            logger.warning(f"Dataset directory {self.dataset_dir} does not exist.")
            return []

        # Recursive scan
        for file_path in self.dataset_dir.rglob("*"):
            if file_path.suffix.lower() in self.loaders:
                loader = self.loaders[file_path.suffix.lower()]
                try:
                    docs = loader.load(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    
        logger.info(f"Total entries loaded: {len(all_documents)}")
        return all_documents

class BaseLoader(ABC):
    """
    Abstract base class for document loaders.
    """
    @abstractmethod
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Loads the content of a file and returns a list of page-level dictionaries.
        """
        pass

class PDFLoader(BaseLoader):
    """
    Implementation of BaseLoader for PDF files using PyMuPDF.
    Handles large files efficiently by iterating through pages.
    """
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        pages_data = []
        doc_id = file_path.stem
        
        try:
            with fitz.open(file_path) as doc:
                for i, page in enumerate(doc):
                    text = page.get_text().strip()
                    if not text:
                        continue
                        
                    pages_data.append({
                        "doc_id": doc_id,
                        "file_name": file_path.name,
                        "page_number": i + 1,
                        "text": text
                    })
            logger.info(f"Loaded {len(pages_data)} pages from {file_path.name}")
        except Exception as e:
            logger.error(f"PyMuPDF error in {file_path.name}: {e}")
            raise
            
        return pages_data

class TextLoader(BaseLoader):
    """
    Implementation of BaseLoader for plain text files.
    Treats the entire file as a single page.
    """
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                
            if not text:
                return []
                
            logger.info(f"Loaded text file: {file_path.name}")
            return [{
                "doc_id": file_path.stem,
                "file_name": file_path.name,
                "page_number": 1,
                "text": text
            }]
        except UnicodeDecodeError:
            logger.error(f"Encoding error in {file_path.name}. Ensure UTF-8.")
            raise
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            raise
