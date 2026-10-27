from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import Chunk, ChunkRequest, DocumentState, EmbedRequest, IndexRequest, QueryRequest, SearchRequest, TextUploadRequest
from .services.chunking import create_chunks
from .services.document_parser import parse_document
from .services.embeddings import embed_query, embed_texts
from .services.llm import REFUSAL_REASON, answer_from_context, is_relevant
from .services.mongo_store import MongoDocumentStore
from .services.vector_store import StoredVector, VectorStore

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "backend" / "uploads"
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Chunking and Embedding Explorer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

documents: dict[str, DocumentState] = {}
vector_store = VectorStore()
mongo_store = MongoDocumentStore()


def api_error(message: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=message)


def save_document(state: DocumentState) -> None:
    mongo_store.save(state.model_dump())


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(documents)}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> DocumentState:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".xlsx", ".xls"}:
        raise api_error("Unsupported file type. Use PDF, DOCX, XLSX, or XLS.")
    started_at = perf_counter()
    started_cpu = process_time()
    process = psutil.Process()
    memory_before = process.memory_info().rss
    read_started = perf_counter()
    content = await file.read()
    read_ms = (perf_counter() - read_started) * 1000
    if not content:
        raise api_error("The uploaded file is empty.")
    document_id = str(uuid4())
    path = UPLOAD_DIR / f"{document_id}{suffix}"
    path.write_bytes(content)
    parse_started = perf_counter()
    try:
        parsed = parse_document(path)
    except Exception as error:
        path.unlink(missing_ok=True)
        if isinstance(error, ValueError):
            raise api_error(str(error)) from error
        raise api_error("The document could not be read. Check that it is not corrupt.") from error
    parse_ms = (perf_counter() - parse_started) * 1000
    total_ms = (perf_counter() - started_at) * 1000
    memory_after = process.memory_info().rss
    parsed["performance"] = {
        "stages_ms": {"file_read": round(read_ms, 2), "document_parsing": round(parse_ms, 2)},
        "ingestion_time_ms": round(total_ms, 2),
        "cpu_time_ms": round((process_time() - started_cpu) * 1000, 2),
        "memory_mb": round(memory_after / (1024 * 1024), 2),
        "memory_delta_mb": round((memory_after - memory_before) / (1024 * 1024), 2),
    }
    state = DocumentState(id=document_id, filename=file.filename or path.name, content_type=file.content_type or "application/octet-stream", file_size=len(content), **parsed)
    documents[document_id] = state
    return state


@app.post("/upload-text")
def upload_text(request: TextUploadRequest) -> DocumentState:
    text = request.text.strip()
    if not text:
        raise api_error("Text cannot be empty.")
    state = DocumentState(
        id=str(uuid4()),
        filename="pasted-text.txt",
        content_type="text/plain",
        file_size=len(text.encode("utf-8")),
        document_type="unstructured",
        classification_reason="Manually entered narrative text is unstructured data.",
        extracted_text=text,
    )
    documents[state.id] = state
    save_document(state)
    return state


@app.post("/chunk")
def chunk_document(request: ChunkRequest) -> DocumentState:
    state = documents.get(request.document_id)
    if not state:
        raise api_error("Document not found.", 404)
    started_at = perf_counter()
    started_cpu = process_time()
    process = psutil.Process()
    memory_before = process.memory_info().rss
    try:
        chunks = create_chunks(state.extracted_text, state.sheets or state.extracted_json)
    except ValueError as error:
        raise api_error(str(error)) from error
    state.chunks = [Chunk(**chunk) for chunk in chunks]
    state.performance.update({
        "stages_ms": {**state.performance.get("stages_ms", {}), "chunking": round((perf_counter() - started_at) * 1000, 2)},
        "ingestion_time_ms": round((perf_counter() - started_at) * 1000, 2),
        "cpu_time_ms": round((process_time() - started_cpu) * 1000, 2),
        "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "memory_delta_mb": round((process.memory_info().rss - memory_before) / (1024 * 1024), 2),
    })
    state.embedding_dimensions = None
    vector_store.clear(state.id)
    save_document(state)
    return state


@app.post("/embed")
def embed_document(request: EmbedRequest) -> DocumentState:
    state = documents.get(request.document_id)
    if not state:
        raise api_error("Document not found.", 404)
    if not state.chunks:
        raise api_error("Create chunks before generating embeddings.")
    started_at = perf_counter()
    started_cpu = process_time()
    process = psutil.Process()
    memory_before = process.memory_info().rss
    try:
        vectors = embed_texts([chunk.text for chunk in state.chunks])
    except Exception as error:
        raise api_error(f"Embedding generation failed: {error}", 500) from error
    for chunk, vector in zip(state.chunks, vectors):
        chunk.embedding = vector
    state.embedding_dimensions = len(vectors[0])
    state.performance.update({
        "stages_ms": {**state.performance.get("stages_ms", {}), "embedding": round((perf_counter() - started_at) * 1000, 2)},
        "cpu_time_ms": round((process_time() - started_cpu) * 1000, 2),
        "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "memory_delta_mb": round((process.memory_info().rss - memory_before) / (1024 * 1024), 2),
    })
    save_document(state)
    return state


@app.post("/index")
def index_document(request: IndexRequest) -> dict:
    state = documents.get(request.document_id)
    if not state:
        raise api_error("Document not found.", 404)
    if not state.chunks or state.embedding_dimensions is None:
        raise api_error("Generate embeddings before indexing.")
    vector_store.add_embeddings(state.id, [StoredVector(chunk.id, chunk.text, chunk.embedding or []) for chunk in state.chunks])
    save_document(state)
    return {"document_id": state.id, "indexed_chunks": len(state.chunks), "dimensions": state.embedding_dimensions, "storage": "MongoDB" if mongo_store.collection is not None else "memory fallback"}


@app.post("/query")
def query_embedding(request: QueryRequest) -> dict:
    try:
        vector = embed_query(request.query)
    except Exception as error:
        raise api_error(f"Query embedding failed: {error}", 500) from error
    return {"query": request.query, "embedding": vector, "dimensions": len(vector)}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    started_at = perf_counter()
    started_cpu = process_time()
    process = psutil.Process()
    memory_before = process.memory_info().rss
    state = documents.get(request.document_id)
    if not state:
        raise api_error("Document not found.", 404)
    if not state.chunks or state.embedding_dimensions is None:
        raise api_error("Generate chunk embeddings before searching.")
    try:
        embedding_started = perf_counter()
        query_vector = embed_query(request.query)
        embedding_ms = (perf_counter() - embedding_started) * 1000
        retrieval_started = perf_counter()
        results = vector_store.search(state.id, query_vector, request.similarity_method, request.top_k)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
    except Exception as error:
        raise api_error(f"Search failed: {error}", 500) from error
    relevance_started = perf_counter()
    for result in results:
        result["method"] = request.similarity_method
    relevant = is_relevant(results)
    relevance_ms = (perf_counter() - relevance_started) * 1000
    generation_started = perf_counter()
    answer = answer_from_context(request.query, results)
    generation_ms = (perf_counter() - generation_started) * 1000
    total_ms = (perf_counter() - started_at) * 1000
    cpu_ms = (process_time() - started_cpu) * 1000
    memory_after = process.memory_info().rss
    performance = {
        "response_time_ms": round(total_ms, 2),
        "ingestion_time_ms": state.performance.get("ingestion_time_ms", 0),
        "cpu_time_ms": round(cpu_ms, 2),
        "cpu_usage_percent": round((cpu_ms / total_ms) * 100, 2) if total_ms else 0,
        "memory_mb": round(memory_after / (1024 * 1024), 2),
        "memory_delta_mb": round((memory_after - memory_before) / (1024 * 1024), 2),
        "chunks_retrieved": len(results),
        "relevance_score": round(results[0]["similarity"], 4) if results else None,
        "relevance_threshold": 0.35 if request.similarity_method != "euclidean" else 1.2,
        "retrieval_speed_chunks_per_second": round(len(results) / (retrieval_ms / 1000), 2) if retrieval_ms else 0,
        "stages_ms": {
            **state.performance.get("stages_ms", {}),
            "query_embedding": round(embedding_ms, 2),
            "vector_search": round(retrieval_ms, 2),
            "relevance_check": round(relevance_ms, 2),
            "response_generation": round(generation_ms, 2),
        },
    }
    return {
        "query": request.query,
        "query_embedding": query_vector,
        "dimensions": len(query_vector),
        "method": request.similarity_method,
        "relevant": relevant,
        "reason": "Retrieved chunks are relevant to the question." if relevant else REFUSAL_REASON,
        "results": results if relevant else [],
        "answer": answer,
        "performance": performance,
    }


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    if document_id not in documents:
        raise api_error("Document not found.", 404)
    documents.pop(document_id)
    vector_store.clear(document_id)
    mongo_store.delete(document_id)
    return {"deleted": document_id}
