# 📄 SYNOPSIS - RAG on My Data

---

## 🔥 Goal

Build a fully **local**, **offline** RAG (Retrieval-Augmented Generation) system on personal files, using:

- Ollama (local LLMs)
- ChromaDB (vector database)
- Moondream (vision-to-text from images)

No external APIs, no cloud services — 100% private.

---

## 🚀 Main Features

- Index and search your personal files:
  - `.txt`, `.csv`, `.md`
  - `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`
  - `.jpg`, `.jpeg`, `.png` (with Moondream)
- Easy file ingestion with `ingest.py`
- Simple CLI search with `search.py`
- Auto-rebuild of the index using `cron`
- Support for local folders, USB drives, and FritzBox network storage
- Access through ChatBox via local Ollama server (`http://<ip>:11434`)

---

## 📂 Project Structure

| File/Folder            | Purpose                                       |
|-------------------------|-----------------------------------------------|
| `install_rag_stack.sh`  | Install all components (Ollama, ChromaDB, etc.) |
| `requirements.txt`      | Python dependencies                           |
| `ingest.py`             | Script to index your files into ChromaDB       |
| `search.py`             | CLI-based search tool                         |
| `cron_example.txt`      | Example of scheduled automatic re-indexing    |
| `rag_data/`             | Folder for your initial data files            |

---

## 🧠 Philosophy

- 📥 Control your own knowledge base
- 🧠 Use AI without leaking data to the cloud
- 🛠️ Adapt it for your specific workflows

---

Built for maximum freedom, privacy, and flexibility. 🚀
