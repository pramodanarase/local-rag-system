# Local RAG System with LlamaIndex and LM Studio

This project implements a Retrieval-Augmented Generation (RAG) system that uses:
- LM Studio for local LLM inference
- Your code/documentation repository as the knowledge base
- LlamaIndex for document processing and retrieval
- Local in-memory vector store

## Setup

1. Install LM Studio from https://lmstudio.ai/
2. Download a suitable model in LM Studio (e.g., Mistral-7B-Instruct or CodeLlama)
3. Start the local inference server in LM Studio (click "Local Inference Server")
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Configure the system:
   - Update `.env` with your LM Studio server settings
   - Specify your repository path in `config.py`

2. Run the indexing:
   ```bash
   python index_repo.py
   ```

3. Query the system:
   ```bash
   python query.py "Your question about the codebase?"
   ```

## How it Works

1. The system indexes your repository's files using LlamaIndex
2. Documents are embedded using SentenceTransformers
3. Queries are processed through a local vector store
4. Relevant context is retrieved and sent to LM Studio's LLM
5. The LLM generates a response based on the retrieved context 