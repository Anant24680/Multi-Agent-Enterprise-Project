from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class ReasoningAgent:
    """Takes document chunks and generates a thoughtful answer to the question."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", temperature: float = 0.1, api_key_manager=None):
        """
        Set up the reasoning agent.
        
        Args:
            model_name: Which Gemini model to use
            temperature: How creative vs focused the responses should be
            api_key_manager: Optional key manager to handle rate limits
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
            ("system", """You're a helpful AI assistant that answers questions based on the documents provided to you.

Here's how to do it:
- Only use information that's actually in the context below
- Point out which chunks you're getting your info from
- If the documents don't have the answer, just say so - don't make stuff up
- Keep your answer clear and to the point, but include all the important details"""),
            ("user", """Question: {question}

Context:
{context}

Please provide:
1. A clear answer to the question
2. Your reasoning (how you got to this answer)
3. Which chunks you used""")
        ])
        
        self.chain = self.prompt | self.llm
    
    def reason(self, query: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Come up with an answer based on the context we found."""
        if not chunks:
            return {
                'answer': "I cannot answer because no relevant documents were found.",
                'reasoning': "No context provided.",
                'chunks_used': [],
                'reasoning_complete': True,
                'error': None
            }
        
        try:
            # Get a fresh API key if we're rotating them
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
            
            # Put together all the context chunks
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                meta = chunk['metadata']
                context_parts.append(
                    f"[Chunk {i}] ({meta['filename']}, Page {meta.get('page', 'N/A')}, "
                    f"Score: {chunk['relevance_score']:.3f})\n{chunk['content']}"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Ask the model for an answer
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
        """Break the response into answer and reasoning parts."""
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


def create_reasoning_agent(model_name: str = "gemini-1.5-flash-latest", api_key_manager=None) -> ReasoningAgent:
    return ReasoningAgent(model_name=model_name, api_key_manager=api_key_manager)
