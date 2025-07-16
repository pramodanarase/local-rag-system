# Local RAG System with Lucene Vector Store

This project implements a Retrieval-Augmented Generation (RAG) system that uses:
- LM Studio for local LLM inference
- Apache Lucene for vector storage and similarity search
- DJL (Deep Java Library) for embeddings generation
- Your code/documentation repository as the knowledge base

## Features

- Efficient vector storage using Apache Lucene
- Persistent index storage with append-only operations
- Batched document processing and indexing
- Semantic search using cosine similarity
- Conversation history support
- Source document tracking

## Setup

1. Install Dependencies:
   - Java 17 or later
   - Maven
   - LM Studio (https://lmstudio.ai/)

2. Configure Environment:
   Create a `.env` file in the project root with:
   ```properties
   # LM Studio settings
   LLAMA_HOST=localhost
   LLAMA_PORT=1234
   
   # Repository settings
   REPO_PATH=/path/to/your/codebase
   EXCLUDE_DIRS=.git,node_modules,__pycache__,venv,.env,target,test
   INCLUDE_EXTENSIONS=.java,.py,.js,.ts,.md
   
   # Index settings
   INDEX_PATH=data/index
   CHUNK_SIZE=256
   CHUNK_OVERLAP=5
   
   # Query settings
   MAX_TOKENS=500
   TEMPERATURE=0.7
   TOP_K=3
   
   # Hugging Face configuration
   HF_TOKEN=your_hf_token
   ```

3. Build the Project:
   ```bash
   mvn clean package
   ```
   This will create two JAR files:
   - `rag-indexer.jar`: For indexing your codebase
   - `rag-query.jar`: For querying the indexed codebase

## Usage

1. Start LM Studio:
   - Open LM Studio
   - Load your preferred model (e.g., Mistral-7B-Instruct or CodeLlama)
   - Start the local inference server

2. Index Your Codebase:
   ```bash
   java -jar target/rag-indexer.jar
   ```
   The indexer will:
   - Process files matching INCLUDE_EXTENSIONS
   - Skip directories in EXCLUDE_DIRS
   - Generate embeddings for text chunks
   - Store vectors and metadata in Lucene index
   - Persist index files in INDEX_PATH

3. Query the System:
   ```bash
   java -jar target/rag-query.jar
   ```
   Features:
   - Interactive query interface
   - Conversation history support
   - Source document display
   - Type 'exit' to quit
   - Type 'clear' to reset conversation history

## How it Works

1. Indexing Process:
   - Files are processed into overlapping chunks
   - Each chunk is embedded using DJL and all-MiniLM-L6-v2
   - Vectors and metadata are stored in Lucene index
   - Index is persisted for future use

2. Query Process:
   - User query is embedded using same model
   - Similar vectors are found using cosine similarity
   - Relevant chunks are retrieved with source info
   - Context is sent to LLM with conversation history
   - LLM generates response based on context

3. Vector Store Features:
   - Append-only operations for data safety
   - Batched commits for performance
   - Normalized vectors for accurate similarity
   - Persistent storage between runs

## Configuration

Key configuration options in `.env`:

- `REPO_PATH`: Path to the codebase to index
- `INDEX_PATH`: Where to store the Lucene index
- `CHUNK_SIZE`: Size of text chunks (in characters)
- `CHUNK_OVERLAP`: Overlap between chunks
- `TOP_K`: Number of similar documents to retrieve
- `TEMPERATURE`: LLM response creativity (0.0-1.0)

## Development

The project uses:
- Java 17
- Maven for build management
- DJL version 0.33.0
- Lucene version 10.2.2
- JUnit 5 for testing

## Notes

- Index files are persistent and append-only
- Large files (>100KB) are skipped
- Default chunk size is 256 characters
- Embeddings use all-MiniLM-L6-v2 model
- Vector similarity uses cosine distance 