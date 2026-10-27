import pytest

from services.embeddings import embed_texts


@pytest.mark.integration
def test_real_embedding_has_dimensions():
    vector = embed_texts(["A short test sentence."])[0]
    assert len(vector) > 0
    assert all(isinstance(value, float) for value in vector)
