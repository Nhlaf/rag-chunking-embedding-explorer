from services.similarity import cosine_similarity, dot_product, euclidean_distance


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0


def test_dot_product():
    assert dot_product([1, 2], [3, 4]) == 11.0


def test_euclidean_distance():
    assert euclidean_distance([1, 0], [0, 1]) == 2**0.5
