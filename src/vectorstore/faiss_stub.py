"""Lightweight FAISS stub using NumPy for environments without faiss-cpu."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


class IndexFlatL2:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:  # pragma: no cover - simple passthrough
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape[1] != self.dimension:
            raise ValueError("Vector dimension mismatch")
        self.vectors = np.vstack([self.vectors, array])

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_array = np.asarray(query, dtype=np.float32)
        distances = np.sum((self.vectors[None, :, :] - query_array[:, None, :]) ** 2, axis=2)
        indices = np.argsort(distances, axis=1)[:, :k]
        ordered_distances = np.take_along_axis(distances, indices, axis=1)
        return ordered_distances, indices.astype("int64")


def write_index(index: IndexFlatL2, path: str | Path) -> None:
    with open(path, "wb") as file:
        pickle.dump(index, file)


def read_index(path: str | Path) -> IndexFlatL2:
    with open(path, "rb") as file:
        return pickle.load(file)
