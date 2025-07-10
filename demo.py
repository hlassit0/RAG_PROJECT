from simple_rag import SimpleRAG
from document_loader import chunk_text

def main():
    # Sample Knowledge Base
    knowledge_base = [
        "RAG (Retrieval-Augmented Generation) is an AI framework that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them as context to generate more accurate and informed responses.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data.",
        "Python is a high-level, interpreted programming language known for its simplicity and versatility. It's widely used in data science, web development, and artificial intelligence applications.",
        "Vector databases store and index high-dimensional vectors, enabling efficient similarity search. They're essential for RAG systems as they allow quick retrieval of semantically similar documents.",
        "Large Language Models (LLMs) like GPT are neural networks trained on vast amounts of text data. They can understand and generate human-like text for various applications."
    ]
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = SimpleRAG()
    
    # Add documents to the RAG system
    all_chunks = []
    for doc in knowledge_base:
        chunks = chunk_text(doc, chunk_size = 200, overlap = 20)
        all_chunks.extend(chunks)
        
    rag.add_documents(all_chunks)
    print(f"Added {len(all_chunks)} document chunks to the knowledge base.")
    
    #Interactive demo
    print("\n=== RAG DEMO ===")
    print("Ask question about AI,ML, or RAG. Type 'exit' to quit.")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        print("\nSearching knowledge base...")
        result = rag.query(question, k=2)
        
        print(f"\nAnswer: {result['answer']}")
        print("\nRelevant Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:100]}... (Score: N/A)")
            
if __name__ == "__main__":
    main()
# This will run the demo, allowing users to ask questions and get answers based on the knowledge base.
# Make sure to have the required packages installed and the environment variable set for OpenAI API key.
# You can add more documents to the `knowledge_base` list to enhance the system's knowledge.
# The `chunk_text` function will split long documents into manageable chunks for better retrieval performance.
# The demo will continue until the user types 'exit' or 'quit'.
# Ensure you have the OpenAI API key set in your environment variables as specified in the `.env.example` file.
# This code provides a simple interactive interface to test the RAG system with your own documents.
# You can modify the `knowledge_base` list to include any text documents you want to use for the demo.
# The system will retrieve relevant chunks and generate answers based on the provided question.
# The results will include both the answer and the sources used to generate that answer,
# allowing users to see where the information came from.        
            