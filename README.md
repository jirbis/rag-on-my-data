# Local RAG Vector on macOS (No GPU, No Torch)

Torch-free local pipeline to index PDF/HTML into a Chroma vector store using `fastembed`, and query it on a Mac without GPU.

## Features
- Recursive ingestion of `.pdf`, `.html`, `.htm`
- Clean HTML extraction (`trafilatura` + BeautifulSoup)
- PDF parsing via `pdfminer.six` with `pypdf` fallback
- Lightweight chunking (no LangChain)
- Embeddings with `fastembed` (ONNX/CPU, no PyTorch)
- Chroma persistent vector DB
- Search with filters (by extension, filename, top-level folder)

## Quick Start

```bash
# create project folder
mkdir -p ~/rag-local-mac && cd ~/rag-local-mac

# create files from this README (copy content into matching filenames)

# Python env
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# (Optional, improves parsing)
brew install tesseract poppler libmagic

# Put your PDFs/HTMLs into ./data (subfolders supported)
export EMB_MODEL="BAAI/bge-small-en-v1.5"  # or any supported fastembed model
export OMP_NUM_THREADS=4 CHUNK_SIZE=800 CHUNK_OVERLAP=120 ADD_BATCH_SIZE=800 EMB_BATCH=128

# Build the vector store
python ingest.py ./data ./vectordb

# Query
python search.py ./vectordb "Find the licensing section" -k 8 --ext .html --name license
