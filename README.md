# RAG Project

A simple RAG (Retrieval-Augmented Generation) implementation with three options:

- **BASIC**: No dependencies, costs nothing (`simple_text_rag.py`)
- **LOCAL**: Uses Hugging Face, works offline (`simple_rag_local.py`) 
- **OPENAI**: Full setup, costs money (`simple_rag.py`)

## Quick Start

### Basic (Recommended)
```bash
python simple_text_rag.py
```
No setup needed. Just works.

### Local 
```bash
pip install -r requirements.txt
python demo_local.py
```

### OpenAI
```bash
pip install -r requirements.txt
# Add OPENAI_API_KEY to .env file
python demo.py
```

## What's What

- `simple_text_rag.py` - Basic version, no dependencies
- `simple_rag_local.py` - Local version with better search
- `simple_rag.py` - OpenAI version (needs API key)
- `demo.py` and `demo_local.py` - Interactive demos
- `document_loader.py` - Text chunking helper

That's it.
echo "OPENAI_API_KEY=your_key_here" > .env
python demo.py
```

##  Quick Start

### 1. Text-Only RAG (No Dependencies)
```bash
python simple_text_rag.py
```

### 2. Interactive Local Demo
```bash
python demo_local.py
```
Then ask questions like:
- "What is Python?"
- "How does RAG work?"
- "Tell me about machine learning"

### 3. Programmatic Usage
```python
from simple_text_rag import SimpleTextRAG

# Initialize RAG system
rag = SimpleTextRAG()

# Add your documents
documents = [
    "Python is a programming language...",
    "Machine learning is a subset of AI...",
    "RAG combines retrieval and generation..."
]
rag.add_documents(documents)

# Ask questions
result = rag.query("What is Python?")
print(result["answer"])
print(result["sources"])
```

## Available RAG Systems

| System | File | Dependencies | Use Case |
|--------|------|-------------|----------|
| **Text RAG** | `simple_text_rag.py` | None | Learning, quick testing |
| **Local RAG** | `simple_rag_local.py` | sentence-transformers, faiss | Offline production use |
| **OpenAI RAG** | `simple_rag.py` | openai, sentence-transformers | High-quality responses |

##  Configuration

### Environment Variables
Create a `.env` file for OpenAI integration:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Customization Options
```python
# Change embedding model
rag = SimpleRAG(model_name='all-mpnet-base-v2')

# Adjust retrieval count
result = rag.query("question", k=5)  # Get top 5 documents

# Modify document chunking
chunks = chunk_text(document, chunk_size=300, overlap=50)
```

##  How RAG Works

1. **Document Ingestion**: Add text documents to the knowledge base
2. **Embedding Creation**: Convert documents to vector representations
3. **Query Processing**: Convert user questions to vectors
4. **Retrieval**: Find most similar documents using cosine similarity
5. **Generation**: Create answers using retrieved context

##  Example Use Cases

- **Customer Support**: Answer questions from company documentation
- **Research Assistant**: Query academic papers and reports
- **Code Documentation**: Search through technical documentation
- **Personal Knowledge Base**: Organize and query personal notes

##  Demo Examples

The interactive demos come with pre-loaded knowledge about:
- Python programming
- Machine learning concepts
- RAG systems
- Natural language processing
- AI and deep learning

##  Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Install missing packages
pip install sentence-transformers faiss-cpu
```

**Memory Issues:**
```python
# Use smaller embedding models
rag = SimpleRAG(model_name='all-MiniLM-L6-v2')
```

**No Results Found:**
```python
# Check if documents were added
print(f"Documents loaded: {len(rag.documents)}")
```

### Performance Tips

- Use `simple_text_rag.py` for fastest startup
- Chunk large documents into smaller pieces (200-500 words)
- Use local models for privacy-sensitive data
- Cache embeddings for frequently used documents

## Learning Resources

- [RAG Paper](https://arxiv.org/abs/2005.11401) - Original research
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [FAISS Documentation](https://faiss.ai/) - Vector similarity search


## Version History

- **v1.0**: Basic RAG implementation with OpenAI
- **v1.1**: Added local RAG option
- **v1.2**: Text-only RAG for zero dependencies
- **v1.3**: Interactive demos and improved documentation

