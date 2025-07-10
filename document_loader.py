import os
from typing import List

def load_text_files(directory: str) -> List[str]:
    """Load all text files from a directory"""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    documents.append(content)
    return documents

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size // 2:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
        
    return [chunk for chunk in chunks if chunk]  # Filter out empty chunks