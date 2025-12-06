from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping


class Distance(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"


@dataclass
class VectorParams:
    size: int
    distance: Distance


@dataclass
class PointStruct:
    id: int
    vector: List[float]
    payload: Mapping[str, Any]


class StubQdrantClient:
    """Minimal client stub used when qdrant-client is unavailable."""

    def __init__(self, url: str, api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        self.recreated: list[tuple[str, VectorParams]] = []
        self.upserts: list[tuple[str, List[PointStruct]]] = []

    def recreate_collection(self, collection_name: str, vectors_config: VectorParams) -> None:
        self.recreated.append((collection_name, vectors_config))

    def upsert(self, collection_name: str, points: List[PointStruct]) -> None:
        self.upserts.append((collection_name, points))
