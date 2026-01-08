import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.models.schemas import DocumentChunk


class VectorStore:
    """Handles document storage and similarity search using FAISS."""
    
    def __init__(self, index_path: str = "app/db/faiss_index", api_key_manager=None, google_api_key: Optional[str] = None):
        """
        Set up the vector store.
        
        Args:
            index_path: Where to save/load the FAISS index
            api_key_manager: Key manager for handling rate limits (recommended)
            google_api_key: Single API key if not using the manager
        """
        self.index_path = index_path
        self.api_key_manager = api_key_manager
        
        # Use the key manager if we have one, otherwise just use a single key
        if api_key_manager:
            current_key = api_key_manager.get_next_key()
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=current_key
            )
        else:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=google_api_key
            )
        
        self.vector_store: Optional[FAISS] = None
        
        # Try to load an existing index if it's there
        if os.path.exists(index_path):
            try:
                self.load()
                print(f"✓ Loaded index from {index_path}")
            except Exception as e:
                print(f"⚠ Could not load index: {e}")
                self.vector_store = None
    
    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the search index."""
        # Get a fresh key if we're rotating
        if self.api_key_manager:
            try:
                current_key = self.api_key_manager.get_next_key()
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=current_key
                )
            except RuntimeError as e:
                print(f"❌ Key rotation error: {e}")
                raise
        
        # Convert our chunks into LangChain's document format
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
        
        # Either create a new index or add to the existing one
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        return len(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Find documents similar to the query."""
        if self.vector_store is None:
            return []
        
        # Get a fresh key if we're rotating them
        if self.api_key_manager:
            try:
                current_key = self.api_key_manager.get_next_key()
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=current_key
                )
            except RuntimeError as e:
                print(f"❌ Key rotation error: {e}")
                raise
        
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
        """Save the index to disk so we don't lose it."""
        if self.vector_store:
            os.makedirs(self.index_path, exist_ok=True)
            self.vector_store.save_local(self.index_path)
    
    def load(self):
        """Load a previously saved index from disk."""
        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get some basic stats about the index."""
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


def create_vector_store(index_path: str = "app/db/faiss_index", api_key_manager=None, google_api_key: Optional[str] = None) -> VectorStore:
    return VectorStore(index_path, api_key_manager, google_api_key)
