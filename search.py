#!/usr/bin/env python3
import os, sys, argparse, re
import chromadb
import numpy as np
from fastembed import TextEmbedding

os.environ.setdefault("OMP_NUM_THREADS", "4")

def color(s):  # простая подсветка терминальной выдачи
    return f"\033[1m{s}\033[0m"

def highlight(snippet, query):
    # выделяем кусочки слов из запроса
    words = [w for w in re.split(r"\W+", query) if len(w) > 2]
    out = snippet
    for w in set(words):
        out = re.sub(rf"(?i)({re.escape(w)})", r"\033[93m\1\033[0m", out)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("db_dir")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--model", default=os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5"))
    p.add_argument("--ext", default=None, help="Filter by extension, e.g. .pdf or .html")
    p.add_argument("--name", default=None, help="Filter by source_name substring")
    args = p.parse_args()

    client = chromadb.PersistentClient(path=args.db_dir)
    coll = client.get_or_create_collection(name="docs", metadata={"hnsw:space": "cosine"})

    # автовыбор поддерживаемой модели fastembed
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    requested = args.model
    if requested not in supported:
        candidates = [
            "intfloat/multilingual-e5-large",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "intfloat/e5-base-v2",
        ]
        requested = next((c for c in candidates if c in supported), next(iter(supported)))
        print(f"[INFO] Модель не поддерживается. Использую: {requested}")

    emb = TextEmbedding(model_name=requested)
    q_vec = np.array(list(emb.query_embed([args.query]))[0], dtype="float32").reshape(1, -1)

    # where-фильтр по метаданным
    where = {}
    if args.ext:
        where["ext"] = args.ext
    if args.name:
        where["source_name"] = {"$contains": args.name}

    res = coll.query(query_embeddings=q_vec, n_results=args.k, where=where if where else None)

    ids      = res.get("ids", [[]])[0]
    docs     = res.get("documents", [[]])[0]
    metas    = res.get("metadatas", [[]])[0]
    dists    = res.get("distances", [[]])[0]

    for i, (rid, doc, meta, d) in enumerate(zip(ids, docs, metas, dists), 1):
        print("="*90)
        print(f"{color(f'Rank {i}')} | distance={d:.4f}")
        print(f"Source: {meta.get('source_name')} [chunk {meta.get('chunk_index')}]")
        print(f"Path:   {meta.get('source_path')}")
        print("-"*90)
        snippet = (doc or "")[:1200].replace("\n", " ")
        print(highlight(snippet, args.query) + ("..." if doc and len(doc) > 1200 else ""))
    print("="*90)

if __name__ == "__main__":
    main()

