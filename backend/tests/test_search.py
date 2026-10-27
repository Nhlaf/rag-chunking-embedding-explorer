from services.vector_store import StoredVector, VectorStore


def test_search_ranks_closest_vector_first():
    store = VectorStore()
    store.add_embeddings("doc", [StoredVector(1, "near", [1, 0]), StoredVector(2, "far", [0, 1])])
    results = store.search("doc", [1, 0], "cosine", 2)
    assert results[0]["chunk_id"] == 1


def test_euclidean_search_ranks_smallest_distance_first():
    store = VectorStore()
    store.add_embeddings("doc", [StoredVector(1, "near", [1, 0]), StoredVector(2, "far", [4, 0])])
    results = store.search("doc", [1, 0], "euclidean", 2)
    assert results[0]["chunk_id"] == 1


def test_irrelevant_similarity_can_be_rejected():
    from services.llm import is_relevant

    assert not is_relevant([{"similarity": 0.1, "method": "cosine"}])
