from functools import lru_cache


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("There is no text to embed.")
    vectors = get_model().encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    return embed_texts([query])[0]
