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
### create project folder
mkdir -p ~/rag-local-mac && cd ~/rag-local-mac
```

#### create files from this README (copy content into matching filenames)

```bash
# Python env
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### (Optional, improves parsing)
```bash
brew install tesseract poppler libmagic
```


### Find supported AI models
```bash
python list_models.py
```


```bash
# Example output
Supported fastembed models:
- BAAI/bge-small-en-v1.5 (384 dimensions)
- BAAI/bge-base-en-v1.5 (768 dimensions)
- intfloat/e5-base-v2 (768 dimensions)
- intfloat/multilingual-e5-large (1024 dimensions)
- BAAI/bge-m3 (1024 dimensions)
```

### Put your PDFs/HTMLs into ./data (subfolders supported)
```bash

export EMB_MODEL="BAAI/bge-small-en-v1.5"  # or any supported fastembed model
export OMP_NUM_THREADS=4 CHUNK_SIZE=800 CHUNK_OVERLAP=120 ADD_BATCH_SIZE=800 EMB_BATCH=128
```

### Build the vector store
```bash
python ingest.py ./data ./vectordb
```

### Query
```bash

python search.py ./vectordb "Find the licensing section" -k 8 --ext .html --name license
```

### Run a Local RAG Search Server

You can start a lightweight Flask-based HTTP server that allows you to:
- Run searches against your vector database (vectordb by default)
- Choose between JSON or HTML output
- Trigger ingestion of new files from the web form

##### Start the server
```bash

python server.py
```


By default, it runs on http://127.0.0.1:8000.

Example request (JSON output)
```bash
curl "http://127.0.0.1:5000/search?query=Huna&output=json"
```

Example request (HTML output)

Open in browser:
```bash
http://127.0.0.1:5000/search?query=Huna&output=html
```

Web Form Interface

Navigate to:

http://127.0.0.1:5000/


From there you can:

Enter a search query

Select output format (JSON or HTML)

Choose model, DB directory, and collection (with defaults ./vectordb and docs)

Trigger ingest.py directly from the form
