# Multi-Agent Knowledge System
**Production-grade AI for trustworthy document Q&A**

An AI-powered system that answers questions from documents **with verified citations and hallucination control**, built using a **multi-agent architecture** inspired by real enterprise RAG systems.

This project demonstrates end-to-end AI engineering skills: system design, agent orchestration, vector search, API development, and deployment.

---

## What this project demonstrates

- Ability to design **production-style RAG systems**
- Strong understanding of **LLM limitations and hallucination risks**
- Practical experience with **multi-agent workflows**
- Clean backend architecture with real-world deployment in mind

This is not a notebook experiment.  
It is a deployable AI service.

---

## Key capabilities

- Document-based Q&A with source attribution
- Semantic search using vector embeddings
- Multi-agent pipeline:
  - Retrieval Agent selects evidence
  - Reasoning Agent generates grounded answers
  - Verification Agent validates citations and detects hallucinations
- Automated refusal when evidence is insufficient
- REST API with interactive documentation
- Web UI for non-technical users
- Dockerized for deployment

---

## System architecture (high level)

```
PDF Upload
   ↓
Text Extraction + Chunking
   ↓
Embedding Generation
   ↓
FAISS Vector Store
   ↓
User Question
   ↓
Retrieval Agent
   ↓
Reasoning Agent
   ↓
Verification Agent
   ↓
Answer + Citations
```

Each component is modular, testable, and replaceable.

---

## Why this matters in real teams

Most LLM applications fail because they:
- hallucinate confidently
- lack explainability
- cannot be trusted in production

This system explicitly addresses those issues by **verifying evidence before answering**.

---

## Tech stack

- **Backend:** FastAPI
- **AI Frameworks:** LangChain, LangGraph
- **LLM:** Google Gemini
- **Vector Database:** FAISS
- **Document Processing:** PyMuPDF
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Docker

---

## Quick start

### Install dependencies

```powershell
python -m pip install --only-binary :all: numpy
python -m pip install -r requirements.txt
python -m pip install langchain-community
```

---


### Run locally

```powershell
python -m uvicorn app.main:app --reload
```

Server runs at  
`http://127.0.0.1:8000`

---

## API overview

- `POST /upload` → Upload and index PDFs
- `POST /query` → Ask questions with citations
- `GET /stats` → System metrics

Interactive docs available at `/docs`.

---

## Project structure

```
app/
├── agents/              # Multi-agent logic
├── ingestion/           # Document processing
├── db/                  # Vector storage
├── models/              # Schemas
├── main.py              # API entrypoint

frontend/
└── index.html           # Web UI
```

---

## Engineering highlights

- Agent-based orchestration using LangGraph
- Explicit separation of retrieval, reasoning, and verification
- Defensive prompting to reduce hallucinations
- Low-temperature inference for factual accuracy
- Local vector search for performance and cost control
- Clean API boundaries and typed schemas

---

## Use cases

- Internal knowledge bases
- Policy and compliance Q&A
- Technical documentation search
- Research paper exploration

---
## Live Demo

Frontend: https://anant24680.github.io/Multi-Agent-Enterprise-Project  
Backend API: https://multi-agent-enterprise-project.onrender.com  

