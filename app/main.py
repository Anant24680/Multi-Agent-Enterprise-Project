import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Make sure Windows console can handle UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import QueryRequest, QueryResponse, UploadResponse
from app.db.vector_store import VectorStore
from app.ingestion.loader import DocumentLoader
from app.agents.retrieval import RetrievalAgent
from app.agents.reasoning import ReasoningAgent
from app.agents.verifier import VerificationAgent
from app.agents.workflow import WorkflowOrchestrator
from app.utils.api_key_manager import APIKeyManager

load_dotenv()

app = FastAPI(
    title="Multi-Agent Knowledge System",
    description="AI-powered Q&A with citations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Where we'll store uploaded files
UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "app/db/faiss_index")

# These get set up when the app starts
vector_store: Optional[VectorStore] = None
document_loader: Optional[DocumentLoader] = None
workflow: Optional[WorkflowOrchestrator] = None
api_key_manager: Optional[APIKeyManager] = None


@app.on_event("startup")
async def startup_event():
    global vector_store, document_loader, workflow, api_key_manager
    
    print("🚀 Starting up the system...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Hmm, can't find GOOGLE_API_KEY in your environment!")
        return
    
    # Set up the API key manager to handle rate limits
    try:
        api_key_manager = APIKeyManager.from_env("GOOGLE_API_KEY", cooldown_minutes=5)
    except ValueError as e:
        print(f"❌ Couldn't set up API Key Manager: {e}")
        return
    
    # Get everything ready
    vector_store = VectorStore(
        index_path=FAISS_INDEX_PATH,
        api_key_manager=api_key_manager
    )
    
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    document_loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    retrieval_agent = RetrievalAgent(vector_store)
    reasoning_agent = ReasoningAgent(
        model_name=os.getenv("MODEL_NAME", "gemini-1.5-flash-latest"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        api_key_manager=api_key_manager
    )
    verification_agent = VerificationAgent(
        model_name=os.getenv("MODEL_NAME", "gemini-1.5-flash-latest"),
        api_key_manager=api_key_manager
    )
    
    workflow = WorkflowOrchestrator(retrieval_agent, reasoning_agent, verification_agent)
    
    stats = vector_store.get_stats()
    key_stats = api_key_manager.get_stats()
    print(f"✓ All set! We've got {stats['total_documents']} documents and {key_stats['total_keys']} API keys ready to go.\n")


@app.get("/")
async def root():
    return {
        "name": "Multi-Agent Knowledge System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "POST /upload",
            "query": "POST /query",
            "stats": "GET /stats"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "components": {
            "vector_store": vector_store is not None,
            "document_loader": document_loader is not None,
            "workflow": workflow is not None
        }
    }


@app.get("/stats")
async def get_stats():
    if not vector_store:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    stats = vector_store.get_stats()
    response = {
        "vector_store": stats,
        "upload_directory": str(UPLOAD_DIR),
        "model": os.getenv("MODEL_NAME", "gemini-1.5-flash-latest")
    }
    
    # Include API key stats if we have them
    if api_key_manager:
        response["api_key_rotation"] = api_key_manager.get_stats()
    
    return response


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    if not vector_store or not document_loader:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        # Save the uploaded file to disk
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"\n📄 Processing: {file.filename}")
        
        # Break it into chunks
        chunks = document_loader.process_document(str(file_path))
        print(f"✓ Created {len(chunks)} chunks")
        
        # Add everything to the search index
        vector_store.add_documents(chunks)
        vector_store.save()
        print(f"✓ Added to index\n")
        
        return UploadResponse(
            success=True,
            filename=file.filename,
            chunks_created=len(chunks),
            message=f"Processed {file.filename} into {len(chunks)} chunks"
        )
    
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    if not workflow or not vector_store:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    stats = vector_store.get_stats()
    if stats['total_documents'] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded. Upload documents first."
        )
    
    try:
        print(f"\n💬 Query: {request.question}")
        
        result = workflow.run(
            question=request.question,
            top_k=request.top_k
        )
        
        if result.get('error'):
            raise HTTPException(
                status_code=500,
                detail=f"Workflow error: {result['error']}"
            )
        
        response = QueryResponse(
            answer=result.get('answer', ''),
            reasoning=result.get('reasoning', ''),
            sources=result.get('verified_sources', []),
            verification_status=result.get('verification_status', 'unverified'),
            warnings=result.get('warnings', [])
        )
        
        print(f"✓ Complete: {result.get('verification_status', 'unknown')}\n")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
