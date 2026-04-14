"""
FastAPI Application — PDF RAG Chatbot with LangGraph + LLMOps

API Endpoints:
  POST /upload           → Upload PDF, returns session_id
  POST /ask              → Ask a question (needs session_id)
  GET  /sessions         → List active sessions
  DELETE /session/{id}   → Delete a session
  GET  /history/{id}     → Get chat history for a session
  POST /feedback         → Submit thumbs up/down feedback
  GET  /feedback/{id}    → Get feedback for a session
  GET  /feedback-stats   → Aggregate feedback statistics
  GET  /                 → Serve frontend
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

import rag_engine

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF RAG Chatbot",
    description="Upload a PDF and ask questions using LangGraph hybrid search pipeline + LLMOps.",
    version="2.0.0"
)

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the React build directory if it exists
if os.path.isdir("frontend/dist/assets"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")


# ─── Request/Response Models ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    source_chunks: list
    expanded_query: str
    session_id: str
    graph_metadata: dict = {}



# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the main HTML page from React build."""
    if os.path.exists("frontend/dist/index.html"):
        return FileResponse("frontend/dist/index.html")
    return JSONResponse({"message": "Frontend not built yet. Run 'npm run build' inside frontend/"})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file. It will be:
      1. Extracted (text from each page)
      2. Chunked (split into small pieces)
      3. Indexed (FAISS + BM25 hybrid search via LangGraph)
    Returns a session_id to use for asking questions.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = rag_engine.create_session(pdf_bytes, file.filename)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Ask a question about the uploaded PDF.
    Uses the LangGraph RAG pipeline:
      1. expand_query (pre-retrieval)
      2. retrieve (hybrid BM25 + FAISS)
      3. grade_documents (relevance check)
      4. generate / fallback (conditional)
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_engine.ask_question(request.session_id, request.question)
        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {"sessions": rag_engine.get_sessions()}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free its memory."""
    if rag_engine.delete_session(session_id):
        return {"message": f"Session {session_id} deleted."}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get chat history for a session."""
    history = rag_engine.get_chat_history(session_id)
    return {"history": history, "session_id": session_id}



# ─── Run Server ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤖 PDF RAG Chatbot v2.0 starting (LangGraph + LLMOps)...")
    print("📄 Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins for now (you can restrict this later to just your Vercel URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)