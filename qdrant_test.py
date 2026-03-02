from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Connect to your local Qdrant instance
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "kylas_minilm_optimized"

# 2. Load the embedding model (MUST be the exact one used for ingestion)
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def test_qdrant_retrieval(query_text, top_k=5):
    print(f"\n🔍 Searching DB for: '{query_text}'")
    print("=" * 60)
    
    try:
        # Convert your text query into a vector array
        query_vector = embed_model.encode(query_text).tolist()
        
        # Search Qdrant
        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points
        
        if not search_results:
            print("❌ No results found. Your collection might be empty.")
            return

        # Loop through and print the top 5 hits
        for i, hit in enumerate(search_results, start=1):
            score = hit.score  # Higher score = better semantic match
            
            # Safely grab the text from the payload (handles missing keys)
            text = hit.payload.get('text', '[WARNING: No "text" key found in payload]')
            
            print(f"\n--- 🎯 Match {i} | Match Score: {score:.4f} ---")
            print(text)
            print("-" * 60)
            
    except Exception as e:
        print(f"\nDatabase Error: {e}")
        print("Make sure your Qdrant container is running and the collection exists.")

if __name__ == "__main__":
    print("\n--- Qdrant Vector DB Diagnostic Tool ---")
    while True:
        test_query = input("\nEnter a test query (or type 'exit' to quit): ")
        if test_query.lower() in ['exit', 'quit']:
            break
            
        test_qdrant_retrieval(test_query)






