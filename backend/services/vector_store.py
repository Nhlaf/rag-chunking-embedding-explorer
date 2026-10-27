from dataclasses import dataclass

from .similarity import higher_is_better, score


@dataclass
class StoredVector:
    chunk_id: int
    text: str
    vector: list[float]


class VectorStore:
    def __init__(self) -> None:
        self._documents: dict[str, list[StoredVector]] = {}

    def add_embeddings(self, document_id: str, vectors: list[StoredVector]) -> None:
        self._documents[document_id] = vectors

    def search(self, document_id: str, query_vector: list[float], method: str, top_k: int) -> list[dict]:
        matches = [
            {"chunk_id": item.chunk_id, "text": item.text, "similarity": score(query_vector, item.vector, method)}
            for item in self._documents.get(document_id, [])
        ]
        return sorted(matches, key=lambda item: item["similarity"], reverse=higher_is_better(method))[:top_k]

    def clear(self, document_id: str) -> None:
        self._documents.pop(document_id, None)
