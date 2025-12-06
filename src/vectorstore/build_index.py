from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for constrained envs
    from src.vectorstore import faiss_stub as faiss
try:  # pragma: no cover - optional dependency
    from qdrant_client.http.models import Distance, PointStruct, VectorParams  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when qdrant is unavailable
    from src.vectorstore.qdrant_stub import Distance, PointStruct, VectorParams
import numpy as np
import pandas as pd
from google import genai

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

EmbeddingFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass
class QdrantOptions:
    """Connection details for uploading vectors to Qdrant."""

    url: str
    collection: str
    api_key: str | None = None
    distance: str = "cosine"


def _load_chunks(input_data: Path | pd.DataFrame | Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if isinstance(input_data, Path):
        return pd.read_parquet(input_data)
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    frame = pd.DataFrame(list(input_data))
    return frame


def build_vector_index(
    chunks: Path | Sequence[Mapping[str, object]],
    embedding_fn: EmbeddingFn,
    store_path: Path,
    batch_size: int | None = None,
    qdrant: QdrantOptions | None = None,
    qdrant_client: object | None = None,
) -> tuple[Path, Path]:
    """Build a FAISS index from text chunks and optionally upload to Qdrant.

    Args:
        chunks: Path to a parquet file or a sequence of mapping objects with a
            ``text`` field and arbitrary metadata.
        embedding_fn: Callable that accepts a batch of strings and returns a
            corresponding batch of embedding vectors.
        store_path: Directory where the index and metadata will be written.
        batch_size: Optional batch size override.
        qdrant: Connection details for Qdrant. When provided, vectors and
            payloads are upserted to the given collection.
        qdrant_client: Optional client override, useful for testing.

    Returns:
        A tuple of paths for the saved FAISS index file and metadata file.
    """

    texts, metadata, matrix = _prepare_embeddings(chunks, embedding_fn, batch_size)

    dimension = int(matrix.shape[1])

    index = faiss.IndexFlatL2(dimension)
    index.add(matrix)

    store_path.mkdir(parents=True, exist_ok=True)
    index_path = store_path / "index.faiss"
    faiss.write_index(index, str(index_path))

    metadata_records = []
    for text, meta in zip(texts, metadata):
        record = {"text": text}
        record.update(meta)
        metadata_records.append(record)

    metadata_path = store_path / "metadata.parquet"
    metadata_frame = pd.DataFrame(metadata_records)
    try:
        metadata_frame.to_parquet(metadata_path, index=False)
    except ImportError:
        metadata_path = store_path / "metadata.jsonl"
        metadata_frame.to_json(metadata_path, orient="records", lines=True, force_ascii=False)

    if qdrant is not None:
        _upsert_qdrant(qdrant, matrix, texts, metadata, client_override=qdrant_client)

    return index_path, metadata_path


def _prepare_embeddings(
    chunks: Path | Sequence[Mapping[str, object]],
    embedding_fn: EmbeddingFn,
    batch_size: int | None,
) -> tuple[List[str], List[Mapping[str, object]], np.ndarray]:
    frame = _load_chunks(chunks)
    if "text" not in frame.columns:
        raise ValueError("Chunks must include a 'text' column")

    texts: List[str] = frame["text"].astype(str).tolist()
    metadata_frame = frame.drop(columns=["text"])
    if metadata_frame.shape[1] == 0:
        metadata = [{} for _ in texts]
    else:
        metadata = metadata_frame.to_dict(orient="records")

    effective_batch = batch_size or BATCH_SIZE
    embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), effective_batch):
        batch = texts[start : start + effective_batch]
        vectors = embedding_fn(batch)
        batch_array = np.asarray(vectors, dtype="float32")
        embeddings.append(batch_array)

    if not embeddings:
        raise ValueError("No embeddings generated; check input data")

    matrix = np.vstack(embeddings)
    return texts, metadata, matrix


def _distance_lookup(name: str):
    normalized = name.lower()
    if normalized == "dot":
        return Distance.DOT
    if normalized in {"l2", "euclid", "euclidean"}:
        return Distance.EUCLID
    return Distance.COSINE


def _get_qdrant_client(url: str, api_key: str | None):
    try:  # pragma: no cover - optional dependency
        from qdrant_client import QdrantClient  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - fallback when qdrant is unavailable
        from src.vectorstore.qdrant_stub import StubQdrantClient as QdrantClient

    return QdrantClient(url=url, api_key=api_key)


def _upsert_qdrant(
    options: QdrantOptions,
    matrix: np.ndarray,
    texts: List[str],
    metadata: List[Mapping[str, object]],
    client_override: object | None = None,
) -> None:
    client = _get_qdrant_client(options.url, options.api_key)
    if client_override is not None:
        client = client_override

    vectors_config = VectorParams(size=matrix.shape[1], distance=_distance_lookup(options.distance))
    client.recreate_collection(collection_name=options.collection, vectors_config=vectors_config)

    points = []
    for idx, (vector, text, meta) in enumerate(zip(matrix.tolist(), texts, metadata)):
        payload = {"text": text, **meta}
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    client.upsert(collection_name=options.collection, points=points)


def _google_embedding_fn(model: str) -> EmbeddingFn:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def embed(batch: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: List[Sequence[float]] = []
        for text in batch:
            response = client.models.embed_content(model=model, content=text)
            if hasattr(response, "values"):
                vectors.append(response.values)
            elif hasattr(response, "embedding"):
                vectors.append(response.embedding)
            elif hasattr(response, "embeddings") and response.embeddings:
                vectors.append(response.embeddings[0].values)
            else:
                raise RuntimeError("Unexpected embedding response structure")
        return vectors

    return embed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS vector index from chunks.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/chunks.parquet"),
        help="Input parquet file with text chunks",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/vector"),
        help="Output directory to store FAISS index and metadata",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Embedding model name for Google GenAI",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size used when computing embeddings",
    )
    parser.add_argument("--qdrant-url", type=str, help="Qdrant endpoint for uploading vectors")
    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default="medical_chunks",
        help="Qdrant collection name to use when uploading",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=os.getenv("QDRANT_API_KEY"),
        help="Optional API key for Qdrant Cloud or secured deployments",
    )
    parser.add_argument(
        "--qdrant-distance",
        type=str,
        default="cosine",
        choices=["cosine", "dot", "l2", "euclid", "euclidean"],
        help="Vector distance metric for the Qdrant collection",
    )
    args = parser.parse_args()

    embedding_fn = _google_embedding_fn(args.model)

    qdrant_options = None
    if args.qdrant_url:
        qdrant_options = QdrantOptions(
            url=args.qdrant_url,
            collection=args.qdrant_collection,
            api_key=args.qdrant_api_key,
            distance=args.qdrant_distance,
        )

    index_path, metadata_path = build_vector_index(
        chunks=args.input,
        embedding_fn=embedding_fn,
        store_path=args.out,
        batch_size=args.batch_size,
        qdrant=qdrant_options,
    )

    print(f"Saved index to {index_path}")
    print(f"Metadata stored at {metadata_path}")


if __name__ == "__main__":
    main()
