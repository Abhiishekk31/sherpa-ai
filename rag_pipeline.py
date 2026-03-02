import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

qdrant = QdrantClient(url="http://localhost:6333")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


hf_token = os.environ.get("HF_TOKEN")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

client = InferenceClient(api_key=hf_token)
API_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"


if not hf_token:
    raise ValueError("Missing HF_TOKEN. Export it via: export HF_TOKEN='your_token'")

def retrieve_context(query, top_k=3):
    """Fetches the best matching chunks from Qdrant."""
    query_vector = embed_model.encode(query).tolist()
    
    search_results = qdrant.query_points(
        collection_name="kylas_minilm_optimized",
        query=query_vector,
        limit=top_k
    ).points 
    
    contexts = [hit.payload['text'] for hit in search_results]
    return "\n---\n".join(contexts)

def ask_kylas_bot(user_query):
    context = retrieve_context(user_query)
    
    system_prompt = (
        "Role: You are the official Kylas CRM Technical Support Assistant. Your goal is to provide "
        "accurate, concise, and helpful technical guidance based strictly on the documentation provided.\n\n"
        
        "Constraints:\n"
        "1. USE ONLY the provided 'Context' block to answer. Do not use outside knowledge.\n"
        "2. IF THE ANSWER IS NOT IN THE CONTEXT, reply: 'I'm sorry, I don't have enough information in "
        "my current database to answer that. Please contact Kylas support at support@kylas.io.'\n"
        "3. FORMATTING: Use bullet points for steps and bold key terms for readability.\n"
        "4. TONE: Maintain a professional, helpful, and technical tone.\n\n"
        
        f"Context:\n{context}"
    )


    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        max_tokens=400,
        temperature=0.1
    )
    
    return completion.choices[0].message.content


if __name__ == "__main__":
    print("\nKylas Support Chatbot Initialized! (Type 'exit' to quit)")
    while True:
        question = input("\nUser: ")
        if question.lower() in ['exit', 'quit']: break
        print(f"\nKylas Bot: {ask_kylas_bot(question)}\n")