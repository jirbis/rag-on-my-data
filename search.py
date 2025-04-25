import chromadb
import argparse

# Initialize ChromaDB client
client = chromadb.Client()
collection = client.get_or_create_collection(name="rag_collection")

def search_query(query_text, n_results=5):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["metadatas", "documents"]
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search in ChromaDB local index.")
    parser.add_argument("--query", required=True, help="The query text to search for.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return.")

    args = parser.parse_args()

    print(f"🔎 Searching for: {args.query}")
    results = search_query(args.query, n_results=args.top_k)

    if results and results['documents']:
        for idx, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"\nResult {idx+1}:")
            print(f"📄 File: {meta['source']}")
            print(f"📝 Snippet: {doc[:500]}...")  # Only show the first 500 characters
    else:
        print("⚠️ No results found.")
