from typing import Any, Literal

from pydantic import BaseModel, Field


SimilarityMethod = Literal["cosine", "dot", "euclidean"]


class ChunkRequest(BaseModel):
    document_id: str


class EmbedRequest(BaseModel):
    document_id: str


class IndexRequest(BaseModel):
    document_id: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)


class TextUploadRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000000)


class SearchRequest(QueryRequest):
    document_id: str
    similarity_method: SimilarityMethod = "cosine"
    top_k: int = Field(default=3, ge=1, le=3)


class Chunk(BaseModel):
    id: int
    text: str
    chunk_type: str = "content"
    character_count: int
    word_count: int
    sentence_count: int
    embedding: list[float] | None = None


class DocumentState(BaseModel):
    id: str
    filename: str
    content_type: str
    file_size: int
    document_type: Literal["structured", "unstructured"]
    classification_reason: str
    extracted_text: str
    extracted_json: list[dict[str, Any]] = Field(default_factory=list)
    sheets: list[dict[str, Any]] = []
    page_count: int | None = None
    chunks: list[Chunk] = Field(default_factory=list)
    embedding_dimensions: int | None = None
    performance: dict[str, Any] = Field(default_factory=dict)
