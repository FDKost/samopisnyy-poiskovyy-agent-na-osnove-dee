from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

from .agent import SimpleVectorStore, DeepAgent

app = FastAPI(title="DeepAgent Search API")

# Initialize store and agent
store = SimpleVectorStore()
agent = DeepAgent(store=store)

# Dummy data for demonstration
sample_docs = [
    {"id": "1", "text": "Python is a programming language that lets you work quickly."},
    {"id": "2", "text": "FastAPI is a modern, fast web framework for building APIs with Python."},
    {"id": "3", "text": "Deep learning models can learn complex patterns from data."},
    {"id": "4", "text": "The quick brown fox jumps over the lazy dog."},
    {"id": "5", "text": "Artificial Intelligence is transforming many industries."},
]
agent.train(sample_docs)

class SearchRequest(BaseModel):
    query: str

class SearchResponseItem(BaseModel):
    id: str
    text: str

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    results = agent.query(request.query)
    return SearchResponse(results=[SearchResponseItem(**r) for r in results])

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
