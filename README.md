# RAG Chunking and Embedding Explorer

A local learning app for exploring document ingestion, embeddings, vector retrieval, grounded answers, and backend performance in a retrieval-augmented generation (RAG) pipeline.

## Features

- Upload PDF, DOCX, XLSX, and XLS documents, or paste text.
- Extract document text and classify content as structured or unstructured.
- Automatically create document-aware chunks for narrative, tabular, and structured content.
- Generate real embeddings locally with `all-MiniLM-L6-v2` and inspect vector dimensions and values.
- Index vectors and search with cosine similarity, dot product, or Euclidean distance.
- Generate answers with Groq and a Qwen model using retrieved document context.
- View backend RAG response time, CPU usage, RAM usage, retrieved chunk count, relevance, retrieval speed, and per-stage timings.
- Persist document JSON to MongoDB when configured; otherwise, use in-memory storage for the running app.

The app makes the RAG ingestion and retrieval pipeline visible:

```text
Document -> Parser -> Text -> Chunker -> Chunks -> Embedding Model
-> Vectors -> Vector Store -> Query Embedding -> Similarity Search -> Top-K Chunks
```

## Technology

- Python 3.10+
- FastAPI and Uvicorn
- Vanilla HTML, CSS, and JavaScript
- `pypdf`, `python-docx`, `openpyxl`, and `xlrd`
- Sentence Transformers and NumPy

## Installation

From the project root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

The first embedding request downloads the sentence-transformer model. Internet access is needed once; later requests use the local cache.

The first embedding call can therefore take 30-60 seconds while the model is downloaded and loaded. Later embedding calls reuse the cached model and are much faster. Chunking itself is deterministic and does not load the embedding model; the model is used only for chunk and query embeddings.

## Configuration

The app can run without external services. To enable answer generation and MongoDB document persistence, set these values in a local `.env` file in the project root (or as environment variables):

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.6-27b
MONGODB_URI=mongodb://127.0.0.1:27017
```

Never commit `.env` or real credentials. The repository ignores `.env` files.

## Start the application

```powershell
python -m uvicorn backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API endpoints

- `GET /health` checks the server.
- `POST /upload` parses a supported document.
- `POST /upload-text` accepts manually entered text.
- `POST /chunk` automatically creates document-aware chunks. It only requires a document id.
- `POST /embed` generates one real vector per chunk.
- `POST /index` stores generated vectors in the in-memory index.
- `POST /query` embeds a query without searching.
- `POST /search` embeds a query and returns ranked chunks.
- `DELETE /documents/{document_id}` removes in-memory document state.

## How the pipeline works

### 1. Document and extraction

The parser turns a file into text. Excel rows are converted into readable statements such as `Employee: John. Department: IT.` before embedding. This preserves the row meaning while making the data suitable for a text embedding model.

### 2. Chunking

A long document is split into smaller pieces because embedding and retrieval work better on focused passages.

- **Structured content:** creates row-oriented chunks from spreadsheet data or numbered document sections so each record can be retrieved independently.
- **Narrative content:** packs paragraphs and sentences into deterministic chunks of up to 1,200 characters, then hard-splits an unusually long sentence when necessary. This keeps chunks bounded without requiring an embedding model during ingestion.

### 3. Embeddings and dimensions

The model maps each chunk to a numeric vector representing semantic meaning. The application measures the returned vector with `len(embedding)`; it does not assume a dimension. With the default model this is commonly 384 dimensions.

```text
Chunk 1 -> [0.02, -0.13, ...] -> N dimensions
Chunk 2 -> [0.11,  0.04, ...] -> N dimensions
```

Every chunk needs its own vector because each chunk can answer a different question.

### 4. Query embedding

The user query is passed through the same model. Comparing vectors created by the same model makes their positions meaningful relative to each other.

### 5. Indexing, similarity, and retrieval

Indexing places each chunk vector into the searchable `VectorStore`. This is the step a real vector database would perform after embeddings are generated.

The explorer offers three comparisons:

- **Cosine similarity:** compares direction or angle. Higher is more similar.
- **Dot product:** multiplies matching dimensions and adds them. Higher is more similar, and vector magnitude matters.
- **Euclidean distance:** measures straight-line distance. Lower is more similar.

Results are ranked using the correct direction for the selected method, then limited by Top-K.

### 6. Vector database later

The current `VectorStore` keeps vectors in memory so the application is easy to run. A production version can replace that class with MongoDB Atlas Vector Search, Qdrant, Chroma, or another vector database while keeping the API flow mostly unchanged.

## MongoDB and JSON

The API response is JSON. For example, after extraction the document has this shape:

```json
{
	"filename": "Introduction to Cloud Computing.docx",
	"document_type": "structured",
	"extracted_json": [{"heading": "2.1 Infrastructure as a Service"}],
	"chunks": []
}
```

After embedding, each chunk also contains an `embedding` array.

When `MONGODB_URI` is configured, document JSON is saved in the `rag_explorer.documents` collection. Vector search currently uses an in-memory index; MongoDB is used for document persistence, not vector search.

For local MongoDB, start MongoDB and set:

```powershell
$env:MONGODB_URI = "mongodb://127.0.0.1:27017"
```

For MongoDB Atlas, set `MONGODB_URI` to the connection string supplied by Atlas. The application falls back to memory if the variable is absent, so the learning app still runs without MongoDB.

## Grounded LLM answers

The application uses Groq with a Qwen model for answer generation. Set your Groq API key before starting the backend:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key"
$env:GROQ_MODEL = "qwen/qwen3.6-27b"
```

The backend sends Groq only the retrieved chunks and tells Qwen to refuse unsupported questions. If the similarity score is too weak, the backend refuses before calling Groq. The embedding model remains local and is independent of the answer-generation provider.

## Testing

Run the fast tests:

```powershell
$env:PYTHONPATH = "backend"
python -m pytest backend/tests -m "not integration" -q
```

Run the real embedding integration test separately:

```powershell
$env:PYTHONPATH = "backend"
python -m pytest backend/tests/test_embeddings.py -m integration -q
```

## Example workflow

1. Upload `company_policy.pdf`.
2. Select `Organize document` and let the app choose meaningful boundaries.
3. Inspect character, word, and sentence counts for the generated chunks.
4. Generate embeddings and expand a vector to see its full representation.
5. Enter `What is the remote work policy?`.
6. Select cosine similarity and Top-K `3`.
7. Inspect the query vector and the three highest-scoring chunks.
