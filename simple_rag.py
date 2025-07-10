import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

class SimpleRAG:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        
        # Initialize local text generation pipeline
        print("Loading text generation model... This may take a moment on first run.")
        self.generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            tokenizer="microsoft/DialoGPT-medium",
            device=-1  # Use CPU
        )

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
        """Generate answer using local Hugging Face model"""
        context = "\n\n".join(context_docs[:2])  # Limit context to avoid token limits
        
        # Create a focused prompt for the local model
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        
        try:
            # Generate response using local model
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,  # Reasonable response length
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                do_sample=True
            )
            
            # Extract the generated text after the prompt
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            # If no meaningful answer generated, provide a simple response
            if not answer or len(answer) < 10:
                answer = f"Based on the context provided, I can see information about: {', '.join([doc[:50] + '...' for doc in context_docs[:2]])}"
            
            return answer
            
        except Exception as e:
            # Fallback: extract relevant sentences from context
            if context_docs:
                return f"Based on the retrieved documents: {context_docs[0][:200]}..."
            return f"I found some relevant information but encountered an error: {str(e)}"
       
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
        
#Example usage
if __name__ == "__main__":
    # Sample documents
    sample_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "RAG systems combine retrieval and generation to provide accurate, contextual answers.",
        "Machine learning algorithms can learn patterns from data without explicit programming."
    ]
    
    # Initialize RAG system
    rag = SimpleRAG()
    rag.add_documents(sample_docs)
    
    # Query the system
    result = rag.query("What is Python?")
    print("Answer:", result["answer"])
    print("\nSources:")
    for doc in result["sources"]:
        print(f"- {doc}")
        
        