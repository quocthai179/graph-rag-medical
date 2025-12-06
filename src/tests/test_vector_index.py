from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback
    from src.vectorstore import faiss_stub as faiss
import numpy as np
import pandas as pd

from src.vectorstore.build_index import QdrantOptions, build_vector_index


def _dummy_embedding(batch: Sequence[str]) -> Sequence[Sequence[float]]:
    vectors = []
    for text in batch:
        # Simple deterministic embedding: length and vowel counts
        length = len(text)
        vowels = sum(text.lower().count(v) for v in "aeiou")
        consonants = length - vowels
        vectors.append([float(length), float(vowels), float(consonants)])
    return vectors


def test_build_index_creates_files_and_search(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"text": "Acute bronchitis affects the bronchi."},
            {"text": "Chronic obstructive pulmonary disease impacts airflow."},
            {"text": "Pneumonia can cause fever and cough."},
        ]
    )

    out_dir = tmp_path / "vector"
    index_path, metadata_path = build_vector_index(frame, _dummy_embedding, out_dir, batch_size=2)

    assert index_path.exists()
    assert metadata_path.exists()

    index = faiss.read_index(str(index_path))
    assert index.ntotal == len(frame)

    query_vector = np.asarray(_dummy_embedding(["bronchitis"]), dtype="float32")
    distances, indices = index.search(query_vector, k=1)
    assert distances.shape == (1, 1)
    assert indices[0][0] in {0, 1, 2}


class _DummyQdrantClient:
    def __init__(self) -> None:
        self.recreated: list[tuple[str, object]] = []
        self.upserts: list[tuple[str, list[object]]] = []

    def recreate_collection(self, collection_name: str, vectors_config: object) -> None:
        self.recreated.append((collection_name, vectors_config))

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.upserts.append((collection_name, points))


def test_build_index_uploads_to_qdrant(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"text": "Hypertension can lead to stroke."},
            {"text": "Type 2 diabetes affects insulin resistance."},
        ]
    )

    dummy_client = _DummyQdrantClient()

    options = QdrantOptions(
        url="http://localhost:6333",
        collection="medical_chunks",
        api_key="token",
        distance="cosine",
    )

    out_dir = tmp_path / "vector"
    build_vector_index(
        frame,
        _dummy_embedding,
        out_dir,
        batch_size=2,
        qdrant=options,
        qdrant_client=dummy_client,
    )

    assert dummy_client.recreated[0][0] == "medical_chunks"
    assert len(dummy_client.upserts[0][1]) == len(frame)
    payloads = [point.payload for point in dummy_client.upserts[0][1]]
    assert any(payload["text"].startswith("Hypertension") for payload in payloads)
