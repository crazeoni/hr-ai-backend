# backend/app.py

from fastapi import FastAPI
from hr_processor.loader import load_hr_document, split_text
from hr_processor.embedder import upsert_documents
from hr_processor.rag_engine import generate_answer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="HR AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://hr-ai-frontend-drab.vercel.app/",
        "https://*.vercel.app"  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # or ["http://localhost:3000"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

class AskRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "running", "service": "HR AI Assistant"}


@app.post("/index-hr")
def index_hr():
    """Loads HR document, splits, and indexes to Pinecone."""
    text = load_hr_document("hr_document.txt")
    docs = split_text(text)
    count = upsert_documents(docs)
    return {"status": "success", "indexed_chunks": count}


@app.post("/ask")
def ask_question(req: AskRequest):
    question = req.question
    result = generate_answer(question)
    return result

# @app.post("/ask")
# def ask_question(question: str):
#     """Ask AI a question about HR policy."""
#     result = generate_answer(question)
#     return result
