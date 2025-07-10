import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re

class SimpleRAGLocal:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        print("Local RAG system initialized (no API key required)")

    def add_documents(self, docs: List[str]):
        """Add documents to the knowledge base"""
        self.documents.extend(docs)
        doc_embeddings = self.encoder.encode(docs)
        if self.embeddings is None:
            self.embeddings = doc_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, doc_embeddings))
        
        # Build Faiss index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k most relevant documents"""
        if self.index is None:
            return []
        
        query_embedding = self.encoder.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
                
        return results
    
    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """Generate answer using simple extractive approach"""
        if not context_docs:
            return "No relevant information found in the knowledge base."
        
        # Simple extractive approach - find most relevant sentences
        query_lower = query.lower()
        best_sentences = []
        
        for doc in context_docs[:2]:  # Use top 2 documents
            sentences = re.split(r'[.!?]+', doc)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Filter out very short sentences
                    # Score based on keyword overlap
                    query_words = set(query_lower.split())
                    sentence_words = set(sentence.lower().split())
                    overlap = len(query_words.intersection(sentence_words))
                    if overlap > 0:
                        best_sentences.append((sentence, overlap))
        
        if best_sentences:
            # Sort by overlap score and take the best
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            return best_sentences[0][0] + "."
        else:
            # If no keyword overlap, return the first document
            return context_docs[0][:200] + "..."
    
    def query(self, question: str, k: int = 3) -> dict:
        """Main RAG query function"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, k)
        
        if not retrieved_docs:
            return {"answer": "No relevant documents found.", "sources": []}
        
        # Extract document texts
        context_docs = [doc for doc, _ in retrieved_docs]
        
        # Generate answer 
        answer = self.generate_answer(question, context_docs)
        
        return {
            "answer": answer,
            "sources": context_docs
        }

# Example usage
if __name__ == "__main__":
    # Sample documents
    sample_docs = [
        "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and AI.",
        "RAG systems combine retrieval and generation to provide accurate, contextual answers. They first find relevant documents, then use them to generate responses.",
        "Machine learning algorithms can learn patterns from data without explicit programming. They improve their performance through experience.",
        "Vector databases store high-dimensional vectors and enable efficient similarity search. They're essential for RAG systems.",
        "Transformers are a type of neural network architecture that revolutionized natural language processing. They use attention mechanisms."
    ]
    
    # Initialize RAG system
    rag = SimpleRAGLocal()
    rag.add_documents(sample_docs)
    
    # Query the system
    result = rag.query("What is Python?")
    print("Answer:", result["answer"])
    print("\nSources:")
    for i, doc in enumerate(result["sources"], 1):
        print(f"{i}. {doc}")
