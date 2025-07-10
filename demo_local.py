from simple_rag_local import SimpleRAGLocal
from document_loader import chunk_text

def main():
    # Sample Knowledge Base
    knowledge_base = [
        "RAG (Retrieval-Augmented Generation) is an AI framework that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them as context to generate more accurate and informed responses.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions.",
        "Python is a high-level, interpreted programming language known for its simplicity and versatility. It's widely used in data science, web development, artificial intelligence applications, and automation tasks.",
        "Vector databases store and index high-dimensional vectors, enabling efficient similarity search. They're essential for RAG systems as they allow quick retrieval of semantically similar documents based on meaning rather than exact keywords.",
        "Large Language Models (LLMs) like GPT are neural networks trained on vast amounts of text data. They can understand and generate human-like text for various applications including chatbots, content creation, and code generation.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. It combines computational linguistics with statistical and machine learning models.",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. It's particularly effective for tasks like image recognition, speech processing, and natural language understanding.",
        "Transformers are a neural network architecture that revolutionized NLP. They use attention mechanisms to process sequential data and form the backbone of modern language models like GPT and BERT."
    ]
    
    # Initialize RAG system (no API key needed!)
    print("Initializing Local RAG system... (No OpenAI API key required)")
    rag = SimpleRAGLocal()
    
    # Add documents to the RAG system
    all_chunks = []
    for doc in knowledge_base:
        chunks = chunk_text(doc, chunk_size=200, overlap=20)
        all_chunks.extend(chunks)
        
    rag.add_documents(all_chunks)
    print(f"Added {len(all_chunks)} document chunks to the knowledge base.")
    
    # Interactive demo
    print("\n=== LOCAL RAG DEMO ===")
    print("Ask questions about AI, ML, Python, or RAG. Type 'exit' to quit.")
    print("Note: This uses local processing - no internet/API required!")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        print("\nSearching local knowledge base...")
        result = rag.query(question, k=3)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nRelevant Sources ({len(result['sources'])} found):")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:100]}...")
            
if __name__ == "__main__":
    main()
