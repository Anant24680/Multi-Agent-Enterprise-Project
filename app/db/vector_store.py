import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.models.schemas import DocumentChunk


class VectorStore:
    """FAISS vector store for document search."""
    
    def __init__(self, index_path: str = "app/db/faiss_index", google_api_key: Optional[str] = None):
        self.index_path = index_path
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        self.vector_store: Optional[FAISS] = None
        
        # Load existing index if available
        if os.path.exists(index_path):
            try:
                self.load()
                print(f"✓ Loaded index from {index_path}")
            except Exception as e:
                print(f"⚠ Could not load index: {e}")
                self.vector_store = None
    
    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the index."""
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    'filename': chunk.filename,
                    'page': chunk.page,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks
                }
            )
            for chunk in chunks
        ]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        return len(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if self.vector_store is None:
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            for doc, score in results
        ]
    
    def save(self):
        """Save the index to disk."""
        if self.vector_store:
            os.makedirs(self.index_path, exist_ok=True)
            self.vector_store.save_local(self.index_path)
    
    def load(self):
        """Load the index from disk."""
        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.vector_store is None:
            return {
                'total_documents': 0,
                'index_exists': False,
                'index_path': self.index_path
            }
        
        return {
            'total_documents': self.vector_store.index.ntotal,
            'index_exists': True,
            'index_path': self.index_path
        }


def create_vector_store(index_path: str = "app/db/faiss_index", google_api_key: Optional[str] = None) -> VectorStore:
    return VectorStore(index_path, google_api_key)
