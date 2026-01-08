from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import Source


class VerificationAgent:
    """Validates answers against sources."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Check if the answer is supported by the sources.

Respond in this format:
VERIFICATION: [verified/partial/unverified]
SUPPORTED_CLAIMS: [what's supported]
UNSUPPORTED_CLAIMS: [what's not supported, or "None"]
WARNINGS: [any concerns, or "None"]
RECOMMENDED_SOURCES: [chunk numbers, e.g., "1, 2, 3"]"""),
            ("user", """Question: {question}
Answer: {answer}
Reasoning: {reasoning}

Sources:
{context}""")
        ])
        
        self.chain = self.prompt | self.llm
    
    def verify(self, query: str, answer: str, reasoning: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Verify answer against sources."""
        if not answer:
            return {
                'verification_status': 'unverified',
                'verified_sources': [],
                'warnings': ['No answer generated'],
                'verification_complete': True,
                'error': None
            }
        
        try:
            # Format context
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                meta = chunk['metadata']
                context_parts.append(
                    f"[Chunk {i}] ({meta['filename']}, Page {meta.get('page', 'N/A')})\n"
                    f"{chunk['content']}"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Verify
            response = self.chain.invoke({
                'question': query,
                'answer': answer,
                'reasoning': reasoning,
                'context': context
            })
            
            result = self._parse_verification(response.content)
            
            # Create source objects
            sources = []
            for idx in result['recommended_sources']:
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    sources.append(Source(
                        content=chunk['content'],
                        filename=chunk['metadata']['filename'],
                        page=chunk['metadata'].get('page'),
                        chunk_index=chunk['metadata']['chunk_index'],
                        relevance_score=chunk['relevance_score']
                    ))
            
            # Use all chunks if none recommended
            if not sources and chunks:
                for chunk in chunks:
                    sources.append(Source(
                        content=chunk['content'],
                        filename=chunk['metadata']['filename'],
                        page=chunk['metadata'].get('page'),
                        chunk_index=chunk['metadata']['chunk_index'],
                        relevance_score=chunk['relevance_score']
                    ))
            
            return {
                'verification_status': result['status'],
                'verified_sources': sources,
                'warnings': result['warnings'],
                'verification_complete': True,
                'error': None
            }
        except Exception as e:
            # On error, return sources but mark unverified
            sources = [
                Source(
                    content=chunk['content'],
                    filename=chunk['metadata']['filename'],
                    page=chunk['metadata'].get('page'),
                    chunk_index=chunk['metadata']['chunk_index'],
                    relevance_score=chunk['relevance_score']
                )
                for chunk in chunks
            ]
            
            return {
                'verification_status': 'unverified',
                'verified_sources': sources,
                'warnings': [f'Verification failed: {str(e)}'],
                'verification_complete': False,
                'error': str(e)
            }
    
    def _parse_verification(self, text: str) -> Dict[str, Any]:
        """Parse verification response."""
        result = {
            'status': 'partial',
            'warnings': [],
            'recommended_sources': []
        }
        
        for line in text.split('\n'):
            line = line.strip()
            
            if line.startswith('VERIFICATION:'):
                status = line.split(':', 1)[1].strip().lower()
                if 'verified' in status and 'unverified' not in status and 'partial' not in status:
                    result['status'] = 'verified'
                elif 'unverified' in status:
                    result['status'] = 'unverified'
            
            elif line.startswith('UNSUPPORTED_CLAIMS:'):
                claims = line.split(':', 1)[1].strip()
                if claims.lower() != 'none':
                    result['warnings'].append(f"Unsupported: {claims}")
            
            elif line.startswith('WARNINGS:'):
                warnings = line.split(':', 1)[1].strip()
                if warnings.lower() != 'none':
                    result['warnings'].append(warnings)
            
            elif line.startswith('RECOMMENDED_SOURCES:'):
                sources = line.split(':', 1)[1].strip()
                try:
                    nums = [int(s.strip()) - 1 for s in sources.split(',') if s.strip().isdigit()]
                    result['recommended_sources'] = nums
                except:
                    pass
        
        return result


def create_verification_agent(model_name: str = "gemini-1.5-flash-latest") -> VerificationAgent:
    return VerificationAgent(model_name=model_name)
