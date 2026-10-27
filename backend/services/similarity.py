import numpy as np


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    return 0.0 if denominator == 0 else float(np.dot(left_vector, right_vector) / denominator)


def dot_product(left: list[float], right: list[float]) -> float:
    return float(np.dot(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))


def euclidean_distance(left: list[float], right: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def score(left: list[float], right: list[float], method: str) -> float:
    if method == "cosine":
        return cosine_similarity(left, right)
    if method == "dot":
        return dot_product(left, right)
    if method == "euclidean":
        return euclidean_distance(left, right)
    raise ValueError(f"Unknown similarity method: {method}")


def higher_is_better(method: str) -> bool:
    return method != "euclidean"
