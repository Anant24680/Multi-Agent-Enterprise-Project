import os
import re
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
import tiktoken
from app.models.schemas import DocumentChunk


class DocumentLoader:
    """Loads and chunks PDF documents."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_document(self, pdf_path: str) -> List[DocumentChunk]:
        """Load PDF and split into chunks."""
        filename = Path(pdf_path).name
        doc = fitz.open(pdf_path)
        
        # Extract text from each page
        pages_text = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages_text.append((page_num, self._clean_text(text)))
        
        doc.close()
        
        # Chunk the text
        all_chunks = []
        for page_num, text in pages_text:
            chunks = self._chunk_text(text, page_num)
            all_chunks.extend(chunks)
        
        # Create DocumentChunk objects
        total_chunks = len(all_chunks)
        return [
            DocumentChunk(
                content=chunk['content'],
                filename=filename,
                page=chunk['page'],
                chunk_index=i,
                total_chunks=total_chunks
            )
            for i, chunk in enumerate(all_chunks)
        ]
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)  # Remove special chars
        return text.strip()
    
    def _chunk_text(self, text: str, page_num: int) -> List[Dict]:
        """Split text into overlapping chunks."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'content': chunk_text,
                'page': page_num
            })
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[DocumentChunk]:
    loader = DocumentLoader(chunk_size, chunk_overlap)
    return loader.process_document(pdf_path)
