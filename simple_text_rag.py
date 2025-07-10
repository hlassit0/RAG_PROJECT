import re
from typing import List, Dict
from collections import Counter
import math

class SimpleTextRAG:
    def __init__(self):
        self.documents = []
        print("Simple Text-based RAG system initialized (no external dependencies)")

    def add_documents(self, docs: List[str]):
        """Add documents to the knowledge base"""
        self.documents.extend(docs)
        print(f"Added {len(docs)} documents. Total: {len(self.documents)} documents")
        
    def preprocess_text(self, text: str) -> List[str]:
        """Simple text preprocessing"""
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def calculate_tf_idf_similarity(self, query: str, document: str) -> float:
        """Calculate TF-IDF based similarity between query and document"""
        query_words = self.preprocess_text(query)
        doc_words = self.preprocess_text(document)
        
        # Simple TF calculation
        query_tf = Counter(query_words)
        doc_tf = Counter(doc_words)
        
        # Calculate similarity based on common words
        common_words = set(query_words) & set(doc_words)
        if not common_words:
            return 0.0
        
        similarity = 0.0
        for word in common_words:
            similarity += query_tf[word] * doc_tf[word]
        
        # Normalize by document length
        doc_length = len(doc_words)
        if doc_length > 0:
            similarity = similarity / math.sqrt(doc_length)
            
        return similarity
        
    def retrieve(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k most relevant documents"""
        if not self.documents:
            return []
        
        print("Searching for relevant documents...")
        
        # Calculate similarity for each document
        doc_scores = []
        for i, doc in enumerate(self.documents):
            score = self.calculate_tf_idf_similarity(query, doc)
            doc_scores.append((doc, score))
        
        # Sort by score and return top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:k]
    
    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """Generate answer using extractive approach"""
        if not context_docs:
            return "No relevant information found."
        
        query_words = set(self.preprocess_text(query))
        best_sentences = []
        
        # Find sentences with most query word overlap
        for doc in context_docs[:2]:
            sentences = re.split(r'[.!?]+', doc)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    sentence_words = set(self.preprocess_text(sentence))
                    overlap = len(query_words & sentence_words)
                    if overlap > 0:
                        best_sentences.append((sentence, overlap))
        
        if best_sentences:
            # Return the sentence with most overlap
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            return best_sentences[0][0] + "."
        else:
            # Return first part of most relevant document
            return context_docs[0][:150] + "..."
    
    def query(self, question: str, k: int = 3) -> Dict:
        """Main RAG query function"""
        retrieved_docs = self.retrieve(question, k)
        
        if not retrieved_docs:
            return {"answer": "No relevant documents found.", "sources": []}
        
        context_docs = [doc for doc, score in retrieved_docs]
        answer = self.generate_answer(question, context_docs)
        
        return {
            "answer": answer,
            "sources": context_docs,
            "scores": [score for doc, score in retrieved_docs]
        }

# Example usage and demo
def main():
    # Sample knowledge base
    knowledge_base = [
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python is widely used in web development, data science, artificial intelligence, and automation.",
        "RAG (Retrieval-Augmented Generation) is an AI framework that combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses those documents as context to generate more accurate and informed responses.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. It combines computational linguistics with statistical models and machine learning techniques.",
        "Vector databases store high-dimensional vectors and enable efficient similarity search. They are essential for modern AI applications, especially RAG systems, as they allow quick retrieval of semantically similar content.",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. It has revolutionized fields like computer vision, speech recognition, and natural language processing.",
        "Transformers are a neural network architecture that revolutionized NLP. They use attention mechanisms to process sequential data and form the backbone of modern language models like GPT, BERT, and ChatGPT."
    ]
    
    # Initialize RAG system
    rag = SimpleTextRAG()
    rag.add_documents(knowledge_base)
    
    print("\n=== SIMPLE TEXT RAG DEMO ===")
    print("Ask questions about Python, AI, ML, or RAG. Type 'exit' to quit.")
    print("Commands:")
    print("  - Ask any question")
    print("  - Type 'add:' followed by text to add to knowledge base")
    print("  - Type 'list' to see all documents")
    print("  - Type 'exit' to quit")
    print("This system works entirely with basic Python - no external APIs needed!")
    
    while True:
        user_input = input("\nYour input: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Check if user wants to add a document
        if user_input.lower().startswith('add:'):
            new_doc = user_input[4:].strip()
            if new_doc:
                rag.add_documents([new_doc])
                print(f" Added document: {new_doc[:60]}...")
            else:
                print(" Please provide text after 'add:'")
            continue
        
        # Check if user wants to list documents
        if user_input.lower() == 'list':
            print(f"\nKnowledge Base ({len(rag.documents)} documents):")
            for i, doc in enumerate(rag.documents, 1):
                print(f"{i}. {doc[:80]}...")
            continue
        
        # Treat as a question
        question = user_input
        result = rag.query(question, k=3)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nFound {len(result['sources'])} relevant sources:")
        for i, (source, score) in enumerate(zip(result['sources'], result['scores']), 1):
            print(f"{i}. {source[:80]}... (relevance: {score:.2f})")

if __name__ == "__main__":
    main()
