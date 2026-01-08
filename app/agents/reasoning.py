from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class ReasoningAgent:
    """Generates answers from retrieved context."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.1):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an AI that answers questions using only the provided documents.

Rules:
- Only use information from the context
- Cite which chunks you used
- If you don't know, say so
- Be concise but thorough"""),
            ("user", """Question: {question}

Context:
{context}

Provide:
1. A direct answer
2. Your reasoning
3. Which chunks you used""")
        ])
        
        self.chain = self.prompt | self.llm
    
    def reason(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer from context."""
        if not chunks:
            return {
                'answer': "I cannot answer because no relevant documents were found.",
                'reasoning': "No context provided.",
                'chunks_used': [],
                'reasoning_complete': True,
                'error': None
            }
        
        try:
            # Format context
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                meta = chunk['metadata']
                context_parts.append(
                    f"[Chunk {i}] ({meta['filename']}, Page {meta.get('page', 'N/A')}, "
                    f"Score: {chunk['relevance_score']:.3f})\n{chunk['content']}"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Get answer
            response = self.chain.invoke({'question': query, 'context': context})
            answer, reasoning = self._parse_response(response.content)
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'chunks_used': list(range(len(chunks))),
                'reasoning_complete': True,
                'error': None
            }
        except Exception as e:
            return {
                'answer': '',
                'reasoning': '',
                'chunks_used': [],
                'reasoning_complete': False,
                'error': f'Reasoning failed: {str(e)}'
            }
    
    def _parse_response(self, text: str) -> tuple[str, str]:
        """Split response into answer and reasoning."""
        lines = text.split('\n')
        answer_lines, reasoning_lines = [], []
        section = 'answer'
        
        for line in lines:
            if 'reasoning' in line.lower() or 'explanation' in line.lower():
                section = 'reasoning'
                continue
            
            if section == 'answer' and line.strip():
                answer_lines.append(line)
            elif section == 'reasoning' and line.strip():
                reasoning_lines.append(line)
        
        answer = '\n'.join(answer_lines).strip() or text
        reasoning = '\n'.join(reasoning_lines).strip() or "Based on the provided context."
        
        return answer, reasoning


def create_reasoning_agent(model_name: str = "gemini-1.5-flash-latest") -> ReasoningAgent:
    return ReasoningAgent(model_name=model_name)
