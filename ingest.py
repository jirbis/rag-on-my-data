import os
import sys
import argparse
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text
from PIL import Image
from mammoth import convert_to_markdown
from pypdf import PdfReader

# Optional: Try to import moondream
try:
    from moondream import Moondream, detect_device
    device = detect_device()
    vision_model = Moondream(device=device)
    has_moondream = True
except ImportError:
    has_moondream = False

# Initialize ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(name="rag_collection")

# Supported file extensions
SUPPORTED_TEXT = [".txt", ".md", ".csv"]
SUPPORTED_DOCS = [".doc", ".docx", ".pdf", ".xls", ".xlsx"]
SUPPORTED_IMAGES = [".jpg", ".jpeg", ".png"]

def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext in SUPPORTED_TEXT:
            with open(filepath, 'r', encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".pdf":
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page
