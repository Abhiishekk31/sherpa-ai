# Kylas Support Chatbot

This repository contains a simple **Retrieval-Augmented Generation (RAG)** pipeline built around a local Qdrant vector database and the Hugging Face Inference API. The chatbot is designed to answer technical support questions for the Kylas CRM product using only the indexed context.

---

## 🚀 Features

- Stores documentation or knowledge in a Qdrant vector collection
- Semantic search using `sentence-transformers` embeddings
- Chat completions via the Hugging Face `meta-llama/Llama-3.1-8B-Instruct` model
- Diagnostic script for validating Qdrant retrievals

## 🧩 Prerequisites

- Python 3.8+ environment
- Local Qdrant instance running on `http://localhost:6333`
- A Hugging Face API token with inference permissions

## 📦 Installation

```bash
python -m pip install -r requirements.txt
```

> You may want to create a virtual environment (venv/conda) before installing.

## 🔧 Configuration

1. **Set your Hugging Face token**:
   ```bash
   export HF_TOKEN="your_token_here"
   ```
2. Ensure Qdrant is running and that the collection `kylas_minilm_optimized` is populated with vectors.
   - The embedding model used for ingestion must match `all-MiniLM-L6-v2`.

## 🧪 Scripts

### `rag_pipeline.py`
This is the main chatbot script. It accepts user input on the command line, retrieves context from Qdrant, and forwards the query to the HF Inference API.

```bash
python rag_pipeline.py
```

Type questions interactively; enter `exit` or `quit` to stop.

### `qdrant_test.py`
A helper tool to manually verify retrieval results from your Qdrant collection.

```bash
python qdrant_test.py
```

Enter any test query and inspect the returned vectors/contexts.

## 📝 Notes

- The system prompt in `rag_pipeline.py` constrains answers strictly to the retrieved context and provides fallback messaging when information is missing.
- Adjust `MODEL_ID` or `top_k` retrieval parameters as needed.

---

## 📧 Contact

For issues or support, contact `support@kylas.io`.
