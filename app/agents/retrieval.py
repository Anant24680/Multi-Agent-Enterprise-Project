from typing import List, Dict, Any
from app.db.vector_store import VectorStore


class RetrievalAgent:
    """Finds relevant documents for a query."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for relevant document chunks."""
        if not query or not query.strip():
            return {
                'chunks': [],
                'query': query,
                'retrieval_complete': False,
                'error': 'Query cannot be empty'
            }
        
        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            
            chunks = [
                {
                    'content': r['content'],
                    'metadata': r['metadata'],
                    'relevance_score': r['score']
                }
                for r in results
            ]
            
            return {
                'chunks': chunks,
                'query': query,
                'retrieval_complete': True,
                'error': None
            }
        except Exception as e:
            return {
                'chunks': [],
                'query': query,
                'retrieval_complete': False,
                'error': f'Retrieval failed: {str(e)}'
            }


def create_retrieval_agent(vector_store: VectorStore) -> RetrievalAgent:
    return RetrievalAgent(vector_store)
