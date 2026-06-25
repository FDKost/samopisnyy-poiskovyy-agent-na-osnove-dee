import json
import random
from typing import List, Dict, Any

class SimpleVectorStore:
    """
    A toy in-memory vector store that stores documents and their embeddings.
    For demonstration purposes only. In production, replace with a proper vector DB.
    """
    def __init__(self):
        self.docs = []  # List[Dict[str, Any]]
        self.embeddings = []  # List[List[float]]

    def add_document(self, doc_id: str, text: str):
        embedding = self._embed(text)
        self.docs.append({"id": doc_id, "text": text})
        self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self._embed(query)
        scores = [self._cosine_similarity(query_emb, emb) for emb in self.embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.docs[i] for i in top_indices]

    def _embed(self, text: str) -> List[float]:
        # Dummy embedding: random vector of length 128
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(128)]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

class DeepAgent:
    """
    Minimal DeepAgent skeleton that uses a SimpleVectorStore for retrieval
    and a placeholder reasoning step.
    """
    def __init__(self, store: SimpleVectorStore):
        self.store = store

    def query(self, question: str) -> List[Dict[str, Any]]:
        # Retrieval
        retrieved = self.store.search(question, top_k=5)
        # Reasoning (placeholder: just return retrieved docs)
        return retrieved

    def train(self, data: List[Dict[str, str]]):
        """
        Dummy training method. In a real scenario, this would fine-tune the
        embedding model and/or the reasoning component.
        """
        for doc in data:
            self.store.add_document(doc_id=doc["id"], text=doc["text"])
