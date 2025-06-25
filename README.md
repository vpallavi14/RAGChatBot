# RAGChatBot
# CrewAI Chatbot with Ollama and RAG

A fully-featured chatbot application that combines:
- CrewAI for agent-based AI processing
- Ollama for local language models 
- RAG (Retrieval-Augmented Generation) for document-based responses
- Clean, responsive user interface

## Enhanced Features

**All original features plus:**
- **Document Intelligence**: Upload PDFs to augment responses with your own knowledge base
- **Context-Aware Responses**: Answers grounded in your uploaded documents
- **Source Tracing**: See which documents contributed to each response
- **Multi-Document Support**: Chat with multiple PDFs simultaneously
- **Persistent Memory**: ChromaDB vector store remembers your documents between sessions

## Prerequisites

- Docker and Docker Compose
- Ollama installed and running ([install guide](https://ollama.com/))
- Minimum 8GB RAM recommended (16GB for larger documents)
- At least 5GB disk space for document storage

## Quick Start

bash
# 1. Clone the repository
git clone https://github.com/yourusername/crewai-chatbot.git
cd crewai-chatbot

# 2. Start Ollama service
ollama serve &

# 3. Pull required models
ollama pull llama2
ollama pull all-minilm  # Required for embeddings
ollama pull mistral    # Optional
ollama pull phi        # Optional
ollama pull gemma      # Optional

# 4. Start the application
docker-compose up --build --force-recreate

## Access the chatbot at:
http://localhost:8000 or open static/index.html directly in your browser

## Usage Guide
# Basic Chat
Type your question in the input box

Press Enter or click Send

Receive an AI-generated response

# Document Intelligence (RAG)
Click the upload button in the settings panel

Select a PDF file (research papers, manuals, etc.)

Toggle "Use RAG" to enable document-based responses

Chat naturally - the AI will reference document content

# Advanced Features
Model Selection: Choose between Llama 2, Mistral, Phi, or Gemma

CrewAI Mode: Enable for multi-agent processing (slower but more thorough)

Hybrid Mode: Combine CrewAI and RAG for maximum intelligence

## Project Structure
text
├── main.py                  # FastAPI backend with RAG integration
├── requirements.txt         # Python dependencies (now with ChromaDB support)
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── chroma_db/               # Vector database storage (auto-created)
└── static/
    ├── index.html           # Enhanced frontend with RAG controls
    └── uploads/             # Temporary PDF storage

## Configuration
# Environment Variables
env
OLLAMA_HOST=http://host.docker.internal:11434  # Ollama connection
CHROMA_DB_PATH=./chroma_db                     # Vector store location
UPLOAD_DIR=./static/uploads                   # PDF upload directory

# Supported File Types
PDF documents (.pdf)


## How It Works
# RAG Pipeline
1. Document Processing:

PDFs are chunked into 1000-character segments

Text embeddings generated using all-MiniLM-L6-v2

Stored in ChromaDB vector database

2. Query Handling:

User question is vectorized

Top 3 most relevant document chunks retrieved

Context is injected into the LLM prompt

3. Response Generation:

Ollama generates answer using document context

Sources are traced back to original PDF chunks

## Troubleshooting
# Document Processing Issues

Ensure PDFs contain selectable text (not scanned images)

Check Docker logs for processing errors

Verify chroma_db directory has write permissions

# Performance Tips

For large documents (>50 pages), increase Docker memory allocation

Restart Ollama if models become unresponsive: ollama serve --restart

Clear document cache by deleting the chroma_db directory

## Pro Tips
# Document Organization:

Upload related documents together for better context

Name files meaningfully (they appear in source references)

# Query Techniques:

"Based on the document, ..." - forces RAG context use

"Compare document A and B on..." - cross-document analysis

# Advanced Configuration:

Adjust chunk size in main.py for different document types

Modify embedding model for non-English documents
