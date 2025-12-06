import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from retrieval.hybrid_retriever import HybridRetriever, RetrieverOutput


class DummyVectorStore:
    def __init__(self, results):
        self.results = results
        self.received_queries = []

    def search(self, query, top_k=5):
        self.received_queries.append((query, top_k))
        return self.results[:top_k]


class DummyGraphClient:
    def __init__(self, k_hop_triples=None, path_triples=None):
        self.k_hop_triples = k_hop_triples or []
        self.path_triples = path_triples or []
        self.expand_calls = []

    def expand_k_hop(self, seeds, k=1):
        self.expand_calls.append(("k_hop", seeds, k))
        return self.k_hop_triples

    def expand_shortest_path(self, seeds, depth=2):
        self.expand_calls.append(("path", seeds, depth))
        return self.path_triples


def test_hybrid_retriever_k_hop_mode():
    vector_results = [
        {"text": "Doc about diabetes", "entity": "Diabetes", "score": 0.9, "source": "doc1"},
        {"text": "Doc about insulin", "entity": "Insulin", "score": 0.8, "source": "doc2"},
        {"text": "Doc about diabetes", "entity": "Diabetes", "score": 0.7, "source": "doc3"},
    ]
    vector_store = DummyVectorStore(vector_results)
    graph_client = DummyGraphClient(k_hop_triples=[("Diabetes", "treated_with", "Insulin")])
    retriever = HybridRetriever(vector_store, graph_client, graph_mode="k_hop", k=2)

    output = retriever.retrieve("diabetes treatment", top_k=2)

    assert isinstance(output, RetrieverOutput)
    assert output.contexts == ["Doc about diabetes", "Doc about insulin"]
    assert output.graph_triples == [("Diabetes", "treated_with", "Insulin")]
    assert output.citations == [
        {"entity": "Diabetes", "score": 0.9, "source": "doc1"},
        {"entity": "Insulin", "score": 0.8, "source": "doc2"},
    ]
    assert vector_store.received_queries == [("diabetes treatment", 2)]
    assert graph_client.expand_calls == [("k_hop", ["Diabetes", "Insulin"], 2)]


def test_hybrid_retriever_path_mode():
    vector_results = [
        {"text": "Doc about hypertension", "entity": "Hypertension", "source": "doc10"},
        {"text": "Doc about stroke risk", "entity": "Stroke", "source": "doc11"},
    ]
    vector_store = DummyVectorStore(vector_results)
    graph_client = DummyGraphClient(path_triples=[("Hypertension", "associated_with", "Stroke")])
    retriever = HybridRetriever(vector_store, graph_client, graph_mode="path", path_depth=3)

    output = retriever.retrieve("stroke causes")

    assert output.contexts == ["Doc about hypertension", "Doc about stroke risk"]
    assert output.graph_triples == [("Hypertension", "associated_with", "Stroke")]
    assert output.citations == [
        {"entity": "Hypertension", "source": "doc10"},
        {"entity": "Stroke", "source": "doc11"},
    ]
    assert graph_client.expand_calls == [("path", ["Hypertension", "Stroke"], 3)]
