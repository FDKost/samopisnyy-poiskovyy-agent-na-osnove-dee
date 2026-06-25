import pytest
from src.agent import SimpleVectorStore, DeepAgent

@pytest.fixture
def agent():
    store = SimpleVectorStore()
    agent = DeepAgent(store=store)
    docs = [
        {"id": "1", "text": "Python is great for data science."},
        {"id": "2", "text": "FastAPI simplifies API development."},
        {"id": "3", "text": "Deep learning models require GPUs."},
    ]
    agent.train(docs)
    return agent

def test_search_returns_results(agent):
    results = agent.query("Python")
    assert len(results) > 0
    assert any("Python" in r["text"] for r in results)

def test_search_empty_query(agent):
    results = agent.query("")
    assert len(results) > 0  # fallback to top docs

def test_store_add_and_search(agent):
    store = agent.store
    store.add_document("4", "Artificial Intelligence is the future.")
    results = store.search("AI")
    assert any("Artificial Intelligence" in r["text"] for r in results)
