from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# What the API expects and returns
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(5, ge=1, le=20)


class Source(BaseModel):
    content: str
    filename: str
    page: Optional[int] = None
    chunk_index: int
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    reasoning: str
    sources: List[Source] = []
    verification_status: str
    warnings: List[str] = []


class UploadResponse(BaseModel):
    success: bool
    filename: str
    chunks_created: int
    message: str


# How agents pass data between each other
class AgentState(BaseModel):
    question: str
    top_k: int = 5
    
    retrieved_chunks: List[Dict[str, Any]] = []
    retrieval_complete: bool = False
    
    answer: str = ""
    reasoning: str = ""
    reasoning_complete: bool = False
    
    verified_sources: List[Source] = []
    verification_status: str = ""
    warnings: List[str] = []
    verification_complete: bool = False
    
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


# How we represent document chunks
class DocumentChunk(BaseModel):
    content: str
    filename: str
    page: Optional[int] = None
    chunk_index: int
    total_chunks: int
    
    class Config:
        arbitrary_types_allowed = True
