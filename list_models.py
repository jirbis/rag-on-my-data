#!/usr/bin/env python3
from fastembed import TextEmbedding

def list_supported_models():
    models = TextEmbedding.list_supported_models()
    print("Supported fastembed models:")
    for m in models:
        print(f"- {m['model']} ({m['dim']} dimensions)")

if __name__ == "__main__":
    list_supported_models()
