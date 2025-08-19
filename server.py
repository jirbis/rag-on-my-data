#!/usr/bin/env python3
import os, re, html, subprocess, threading
import numpy as np
import chromadb
from fastembed import TextEmbedding
from flask import Flask, request, jsonify, Response, redirect, url_for

# ---------------- Defaults ----------------
DEFAULT_DB_DIR   = "./vectordb"
DEFAULT_DATA_DIR = "./data"
DEFAULT_MODEL    = "BAAI/bge-small-en-v1.5"
DEFAULT_COLL     = "docs"

# ---------------- Global state ----------------
#global current_db, current_model, current_coll, client, collection, embedder
  
current_db   = os.getenv("DB_DIR", DEFAULT_DB_DIR)
current_model = os.getenv("EMB_MODEL", DEFAULT_MODEL)
current_coll  = os.getenv("COLLECTION", DEFAULT_COLL)
client         = None
collection     = None
embedder       = None


client = chromadb.PersistentClient(path=current_db)
collection = client.get_or_create_collection(name=current_coll, metadata={"hnsw:space": "cosine"})
embedder = TextEmbedding(model_name=current_model)

app = Flask(__name__)

# ---------------- Helpers ----------------
def build_where(ext: str | None, name: str | None):
    where = {}
    if ext:
        where["ext"] = ext
    if name:
        where["source_name"] = {"$contains": name}
    return where or None

def highlight_text(snippet: str, query: str, mode: str = "html") -> str:
    words = [w for w in re.split(r"\W+", query) if len(w) > 2]
    out = snippet
    for w in set(words):
        if mode == "html":
            out = re.sub(rf"(?i)({re.escape(w)})", r"<mark>\1</mark>", out)
        else:
            out = re.sub(rf"(?i)({re.escape(w)})", r"**\1**", out)
    return out

def query_vectors(q: str, k: int, where):
    q_vec = np.array(list(embedder.query_embed([q]))[0], dtype="float32").reshape(1, -1)
    return collection.query(query_embeddings=q_vec, n_results=k, where=where)

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    """Render input form."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Local RAG Search</title>
<style>
body{{font-family:system-ui,Arial,sans-serif; margin:24px; max-width:960px}}
h1{{margin:0 0 12px}}
.form input, .form select{{margin:4px 0; padding:6px}}
button{{padding:8px 14px; margin-top:6px; cursor:pointer}}
</style></head><body>
<h1>Local RAG Search</h1>
<form class="form" method="get" action="/search" target="_self">
  <label>Query:</label><br>
  <input type="text" name="q" placeholder="Type your query..." value=""><br>
  
  <label>Top K:</label><br>
  <input type="number" name="k" value="5" min="1" max="50"><br>
  
  <label>Format:</label><br>
  <select name="format">
    <option value="html" selected>HTML</option>
    <option value="json">JSON</option>
  </select><br>

  <label>Model:</label><br>
  <input type="text" name="model" value="{html.escape(current_model)}"><br>

  <label>DB_DIR:</label><br>
  <input type="text" name="db" value="{html.escape(current_db)}"><br>

  <label>COLLECTION:</label><br>
  <input type="text" name="coll" value="{html.escape(current_coll)}"><br>

  <button type="submit">Search</button>
</form>

<hr>
<form class="form" method="post" action="/ingest" target="_self">
  <h3>Run ingestion (ingest.py)</h3>
  <label>Input data dir:</label><br>
  <input type="text" name="input_dir" value="{html.escape(DEFAULT_DATA_DIR)}"><br>
  
  <label>DB_DIR:</label><br>
  <input type="text" name="db" value="{html.escape(current_db)}"><br>
  
  <label>COLLECTION:</label><br>
  <input type="text" name="coll" value="{html.escape(current_coll)}"><br>

  <button type="submit">Start ingestion</button>
</form>
</body></html>"""

@app.get("/search")
def search():
    global current_db, current_model, current_coll, client, collection, embedder
    q = (request.args.get("q") or "").strip()
    if not q:
        return redirect(url_for("index"))

    k      = int(request.args.get("k", 5))
    fmt    = (request.args.get("format") or "json").lower().strip()
    model  = request.args.get("model", current_model)
    db     = request.args.get("db", current_db)
    coll   = request.args.get("coll", current_coll)
    ext    = request.args.get("ext")
    name   = request.args.get("name")

    # Switch if model/DB changed
    
    if db != current_db or coll != current_coll or model != current_model:
        current_db, current_model, current_coll = db, model, coll
        client = chromadb.PersistentClient(path=db)
        collection = client.get_or_create_collection(name=coll, metadata={"hnsw:space": "cosine"})
        embedder = TextEmbedding(model_name=model)

    res = query_vectors(q, max(1, k), build_where(ext, name))
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # JSON output
    if fmt == "json":
        results = []
        for i, (rid, doc, meta, d) in enumerate(zip(ids, docs, metas, dists), 1):
            snippet = (doc or "")[:800].replace("\n", " ")
            results.append({
                "rank": i,
                "distance": float(d),
                "source_name": meta.get("source_name"),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "ext": meta.get("ext"),
                "path": meta.get("source_path"),
                "snippet": highlight_text(snippet, q, mode="json") + ("..." if doc and len(doc) > 800 else "")
            })
        return jsonify({"query": q, "k": k, "results": results})

    # HTML output
    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Search Results</title>
<style>
body{{font-family:system-ui,Arial,sans-serif; margin:24px; max-width:960px}}
.result{{border:1px solid #ccc; border-radius:10px; padding:12px; margin:14px 0}}
mark{{background:#ffef7a}}
</style></head><body>
<h2>Results for: {html.escape(q)}</h2>"""]

    for i, (rid, doc, meta, d) in enumerate(zip(ids, docs, metas, dists), 1):
        snippet_raw = (doc or "")[:1000].replace("\n", " ")
        snippet = highlight_text(html.escape(snippet_raw), q, "html")
        src = meta.get("source_name", "")
        path = meta.get("source_path", "")
        chunk = int(meta.get("chunk_index", 0))
        extv = meta.get("ext", "")
        parts.append(f"""
<div class="result">
  <div><strong>Rank {i}</strong> (distance {d:.4f})</div>
  <div>Source: <a href="file://{html.escape(path)}" target="_blank">{html.escape(src)}</a>, chunk {chunk}, ext {html.escape(extv)}</div>
  <div>{snippet}...</div>
</div>""")

    parts.append("</body></html>")
    return Response("".join(parts), mimetype="text/html")

@app.post("/ingest")
def ingest():
    input_dir = request.form.get("input_dir", DEFAULT_DATA_DIR)
    db        = request.form.get("db", current_db)
    coll      = request.form.get("coll", current_coll)

    def run_ingest():
        subprocess.run(["python3", "ingest.py", input_dir, db], check=False)

    threading.Thread(target=run_ingest, daemon=True).start()
    return f"<p>Ingestion started for <b>{html.escape(input_dir)}</b> → DB {html.escape(db)}, collection {html.escape(coll)}</p><a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)

