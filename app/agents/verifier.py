from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import Source


class VerificationAgent:
    """Double-checks that answers are actually supported by the source documents."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.0, api_key_manager=None):
        """
        Set up the verification agent.
        
        Args:
            model_name: Which Gemini model to use
            temperature: Keep at 0 for consistent verification
            api_key_manager: Optional key manager for rate limits
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key_manager = api_key_manager
        
        # Set up the language model
        if api_key_manager:
            current_key = api_key_manager.get_next_key()
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=current_key
            )
        else:
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Your job is to check if the answer is actually backed up by the sources.

Please respond in this format:
VERIFICATION: [verified/partial/unverified]
SUPPORTED_CLAIMS: [what parts are backed up by the sources]
UNSUPPORTED_CLAIMS: [what parts aren't backed up, or write "None"]
WARNINGS: [any concerns you have, or write "None"]
RECOMMENDED_SOURCES: [which chunk numbers support the answer, like "1, 2, 3"]"""),
            ("user", """Question: {question}
Answer: {answer}
Reasoning: {reasoning}

Sources:
{context}""")
        ])
        
        self.chain = self.prompt | self.llm
    
    def verify(self, query: str, answer: str, reasoning: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Check if the answer is actually supported by the sources."""
        if not answer:
            return {
                'verification_status': 'unverified',
                'verified_sources': [],
                'warnings': ['No answer generated'],
                'verification_complete': True,
                'error': None
            }
        
        try:
            # Get a fresh API key if we're rotating
            if self.api_key_manager:
                try:
                    current_key = self.api_key_manager.get_next_key()
                    self.llm = ChatGoogleGenerativeAI(
                        model=self.model_name,
                        temperature=self.temperature,
                        google_api_key=current_key
                    )
                    self.chain = self.prompt | self.llm
                except RuntimeError as e:
                    print(f"❌ Key rotation error: {e}")
                    raise
            
            # Format the context for verification
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                meta = chunk['metadata']
                context_parts.append(
                    f"[Chunk {i}] ({meta['filename']}, Page {meta.get('page', 'N/A')})\n"
                    f"{chunk['content']}"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Run the verification
            response = self.chain.invoke({
                'question': query,
                'answer': answer,
                'reasoning': reasoning,
                'context': context
            })
            
            result = self._parse_verification(response.content)
            
            # Turn the results into proper source objects
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
            
            # If no specific sources were recommended, just use all of them
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
            # If verification fails, still return the sources but mark as unverified
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
        """Parse the verification response from the model."""
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


def create_verification_agent(model_name: str = "gemini-1.5-flash-latest", api_key_manager=None) -> VerificationAgent:
    return VerificationAgent(model_name=model_name, api_key_manager=api_key_manager)
