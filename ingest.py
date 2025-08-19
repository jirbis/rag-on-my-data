#!/usr/bin/env python3
import os, sys, json, hashlib, pathlib, traceback, re
from typing import List, Dict, Iterable, Tuple
from multiprocessing import Pool, cpu_count

# --- Parsers ---
from bs4 import BeautifulSoup
import trafilatura
from pdfminer.high_level import extract_text as pdfminer_extract
from pypdf import PdfReader

# --- Vector DB and embeddings ---
import chromadb
import numpy as np
from fastembed import TextEmbedding

# ---------------- Settings ----------------
INPUT_DIR   = sys.argv[1] if len(sys.argv) > 1 else "data"
DB_DIR      = sys.argv[2] if len(sys.argv) > 2 else "vectordb"
COLLECTION  = "docs"

# FastEmbed model (multilingual or English):
# Examples: "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2", etc.
EMB_MODEL   = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")

# Chunking
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "120"))

# Batching
ADD_BATCH_SIZE  = int(os.getenv("ADD_BATCH_SIZE", "800"))
EMB_BATCH       = int(os.getenv("EMB_BATCH", "128"))

# Limit number of threads (safety)
os.environ.setdefault("OMP_NUM_THREADS", "4")

# --------------- Utilities ----------------
def hash_id(text: str, meta: Dict) -> str:
    """Generate a stable SHA256 hash for text + metadata."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    return h.hexdigest()

def clean_html_with_bs4(raw: str) -> str:
    """Fallback HTML cleaning with BeautifulSoup if trafilatura fails."""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)

def parse_html(path: str) -> str:
    """Extract readable text from HTML using trafilatura or BeautifulSoup fallback."""
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="ignore")
    extracted = trafilatura.extract(raw, include_comments=False, include_tables=True)
    if extracted and extracted.strip():
        return extracted
    return clean_html_with_bs4(raw)

def parse_pdf(path: str) -> str:
    """Extract text from PDF using pdfminer or PyPDF as fallback."""
    try:
        txt = pdfminer_extract(path)
        if txt and txt.strip():
            return txt
    except Exception:
        pass
    try:
        reader = PdfReader(path)
        texts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                texts.append(t)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    return ""

# --- Simple sentence-based splitter ---
_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks while trying to respect sentence boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0

    def flush():
        nonlocal buf, cur_len
        if buf:
            block = " ".join(buf).strip()
            if block:
                chunks.append(block)
            buf, cur_len = [], 0

    for p in paragraphs:
        sentences = [s.strip() for s in _SENT_SPLIT.split(p) if s.strip()]
        for s in sentences:
            s_len = len(s) + 1
            if cur_len + s_len <= chunk_size:
                buf.append(s); cur_len += s_len
            else:
                flush()
                if s_len > chunk_size:
                    # Break long sentence into sliding window chunks
                    i = 0
                    step = chunk_size - overlap if chunk_size > overlap else chunk_size
                    while i < len(s):
                        chunk = s[i:i+chunk_size]
                        chunks.append(chunk)
                        i += step if step > 0 else chunk_size
                    buf, cur_len = [], 0
                else:
                    buf = [s]; cur_len = s_len
        if cur_len >= chunk_size * 0.9:
            flush()
    flush()

    # Merge with overlap
    if overlap > 0 and chunks:
        final = []
        for i, ch in enumerate(chunks):
            if i == 0:
                final.append(ch)
            else:
                prev = final[-1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                merged = (tail + " " + ch).strip()
                if len(merged) > chunk_size:
                    final.append(ch)
                else:
                    final[-1] = merged
        chunks = final
    return chunks

def file_to_chunks(file_path: str) -> List[Tuple[str, str, Dict]]:
    """Parse a single file (PDF/HTML) and return chunks with metadata."""
    ext = pathlib.Path(file_path).suffix.lower()
    try:
        if ext == ".pdf":
            text = parse_pdf(file_path)
        elif ext in {".html", ".htm"}:
            text = parse_html(file_path)
        else:
            return []
        if not text or not text.strip():
            return []

        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        out = []
        for i, ch in enumerate(chunks):
            meta = {
                "source_path": os.path.abspath(file_path),
                "source_name": os.path.basename(file_path),
                "chunk_index": i,
                "ext": ext,
            }
            uid = hash_id(ch, meta)
            out.append((uid, ch, meta))
        return out
    except Exception:
        print(f"[WARN] parse failed: {file_path}\n{traceback.format_exc()}")
        return []

def iter_all_files(root: str) -> List[str]:
    """Recursively list all supported files (PDF, HTML) in directory tree."""
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".pdf", ".html", ".htm")):
                paths.append(os.path.join(dp, fn))
    return paths

def mp_collect_records(paths: List[str]):
    """Parallel parse files into (id, chunk, metadata) tuples."""
    procs = max(1, cpu_count() - 1)
    with Pool(processes=procs) as p:
        for recs in p.imap_unordered(file_to_chunks, paths, chunksize=8):
            for r in recs:
                yield r

def embed_texts(emb_model: TextEmbedding, texts: List[str], batch: int) -> np.ndarray:
    """Embed text in batches with FastEmbed, return numpy array."""
    vecs = []
    start = 0
    while start < len(texts):
        end = min(start + batch, len(texts))
        part = texts[start:end]
        # emb_model.embed() yields a generator
        for v in emb_model.embed(part):
            vecs.append(v)
        start = end
    return np.array(vecs, dtype="float32")

def add_in_batches(collection, emb_model: TextEmbedding, ids, texts, metas):
    """Add records to Chroma collection in batches with pre-computed embeddings."""
    embeddings = embed_texts(emb_model, texts, EMB_BATCH)
    for i in range(0, len(ids), ADD_BATCH_SIZE):
        sl = slice(i, i + ADD_BATCH_SIZE)
        collection.add(
            ids=ids[sl],
            documents=texts[sl],
            metadatas=metas[sl],
            embeddings=embeddings[sl]
        )

def main():
    os.makedirs(DB_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding model (fastembed): {EMB_MODEL}")
    emb_model = TextEmbedding(model_name=EMB_MODEL)

    paths = iter_all_files(INPUT_DIR)
    print(f"Found {len(paths)} files in {INPUT_DIR}")

    ids, texts, metas = [], [], []
    seen = set()

    # Collect and ingest in batches
    for uid, ch, meta in mp_collect_records(paths):
        if uid in seen:
            continue
        seen.add(uid)
        ids.append(uid); texts.append(ch); metas.append(meta)

        if len(ids) >= ADD_BATCH_SIZE * 2:
            print(f"Ingesting batch of {len(ids)} chunks...")
            add_in_batches(collection, emb_model, ids, texts, metas)
            ids, texts, metas = [], [], []

    if ids:
        print(f"Ingesting final {len(ids)} chunks...")
        add_in_batches(collection, emb_model, ids, texts, metas)

    # Quick test query
    q = "Briefly: what is this corpus of documents about?"
    q_vec = np.array(list(emb_model.query_embed([q]))[0], dtype="float32").reshape(1, -1)
    res = collection.query(query_embeddings=q_vec, n_results=3)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
