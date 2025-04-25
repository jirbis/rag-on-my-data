# 📚 RAG on My Data

Your local RAG (Retrieval-Augmented Generation) server powered by Ollama, ChromaDB, and Moondream, running fully offline on your own machine.

---

## 📦 Project structure

- `install_rag_stack.sh` — automatic installation of all necessary components
- `requirements.txt` — Python dependencies
- `ingest.py` — script for indexing files into ChromaDB
- `search.py` — simple CLI search tool
- `cron_example.txt` — example for automatic re-indexing
- `rag_data/` — folder for your initial files

---

## 🛠 Installation

```bash
git clone https://github.com/jirbis/rag-on-my-data.git
cd rag-on-my-data
bash install_rag_stack.sh
source venv/bin/activate
