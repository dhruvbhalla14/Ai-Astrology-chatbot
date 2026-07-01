from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag import chat, ensemble_retriever

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "templates" / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str


@app.get("/")
async def serve_index():
    return FileResponse(INDEX_HTML)


@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    response = chat(request.message)
    
    # fetch sources for the UI
    docs = ensemble_retriever.invoke(request.message)
    sources = [
        {
            "book": d.metadata.get("book", ""),
            "chapter": d.metadata.get("chapter", ""),
            "text": d.page_content[:300]
        }
        for d in docs[:3]  # top 3 sources
    ]

    return {"response": response, "sources": sources}
