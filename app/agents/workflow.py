from typing import TypedDict
from langgraph.graph import StateGraph, END
from app.models.schemas import AgentState
from app.agents.retrieval import RetrievalAgent
from app.agents.reasoning import ReasoningAgent
from app.agents.verifier import VerificationAgent


class WorkflowOrchestrator:
    """Coordinates the multi-agent workflow."""
    
    def __init__(self, retrieval_agent, reasoning_agent, verification_agent):
        self.retrieval_agent = retrieval_agent
        self.reasoning_agent = reasoning_agent
        self.verification_agent = verification_agent
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("verification", self._verification_node)
        
        workflow.set_entry_point("retrieval")
        workflow.add_edge("retrieval", "reasoning")
        workflow.add_edge("reasoning", "verification")
        workflow.add_edge("verification", END)
        
        return workflow.compile()
    
    def _retrieval_node(self, state: AgentState) -> AgentState:
        print("🔍 Searching documents...")
        
        result = self.retrieval_agent.retrieve(state.question, state.top_k)
        
        if result['error']:
            state.error = result['error']
            state.retrieval_complete = False
        else:
            state.retrieved_chunks = result['chunks']
            state.retrieval_complete = True
            print(f"✓ Found {len(state.retrieved_chunks)} chunks")
        
        return state
    
    def _reasoning_node(self, state: AgentState) -> AgentState:
        print("🧠 Generating answer...")
        
        if not state.retrieval_complete or state.error:
            state.answer = "Unable to generate answer due to retrieval failure."
            state.reasoning_complete = False
            return state
        
        result = self.reasoning_agent.reason(state.question, state.retrieved_chunks)
        
        if result['error']:
            state.error = result['error']
            state.reasoning_complete = False
        else:
            state.answer = result['answer']
            state.reasoning = result['reasoning']
            state.reasoning_complete = True
            print(f"✓ Answer generated")
        
        return state
    
    def _verification_node(self, state: AgentState) -> AgentState:
        print("✅ Verifying answer...")
        
        if not state.reasoning_complete or state.error:
            state.verification_status = "unverified"
            state.warnings = ["Previous steps failed"]
            state.verification_complete = False
            return state
        
        result = self.verification_agent.verify(
            state.question,
            state.answer,
            state.reasoning,
            state.retrieved_chunks
        )
        
        if result['error']:
            state.error = result['error']
            state.verification_complete = False
        else:
            state.verified_sources = result['verified_sources']
            state.verification_status = result['verification_status']
            state.warnings = result['warnings']
            state.verification_complete = True
            print(f"✓ Status: {state.verification_status}")
        
        return state
    
    def run(self, question: str, top_k: int = 5) -> AgentState:
        """Run the workflow for a question."""
        print(f"\n{'='*60}")
        print(f"🚀 Query: {question}")
        print(f"{'='*60}\n")
        
        initial_state = AgentState(question=question, top_k=top_k)
        final_state = self.workflow.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"✨ Complete!")
        print(f"{'='*60}\n")
        
        return final_state


def create_workflow(retrieval_agent, reasoning_agent, verification_agent) -> WorkflowOrchestrator:
    return WorkflowOrchestrator(retrieval_agent, reasoning_agent, verification_agent)
