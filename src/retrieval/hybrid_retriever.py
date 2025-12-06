from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass
class RetrieverOutput:
    """Container for retriever results."""

    contexts: List[str]
    graph_triples: List[Tuple[str, str, str]]
    citations: List[Dict[str, Any]]


class HybridRetriever:
    """Simple hybrid retriever combining vector and graph expansion."""

    def __init__(
        self,
        vector_store: Any,
        graph_client: Any,
        *,
        graph_mode: str = "k_hop",
        k: int = 1,
        path_depth: int = 2,
    ) -> None:
        self.vector_store = vector_store
        self.graph_client = graph_client
        self.graph_mode = graph_mode
        self.k = k
        self.path_depth = path_depth

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrieverOutput:
        vector_results = self._search_vector(query, top_k=top_k)
        seed_entities = self._seed_entities(vector_results)
        graph_triples = self._expand_graph(seed_entities)
        contexts, citations = self._rerank_and_merge(vector_results)
        return RetrieverOutput(contexts=contexts, graph_triples=graph_triples, citations=citations)

    def _search_vector(self, query: str, *, top_k: int) -> Sequence[Dict[str, Any]]:
        """Run semantic search against the vector store."""

        return self.vector_store.search(query, top_k=top_k)

    def _seed_entities(self, results: Iterable[Dict[str, Any]]) -> List[str]:
        """Extract unique entity identifiers from vector results."""

        seen = set()
        seeds: List[str] = []
        for result in results:
            entity = result.get("entity")
            if entity and entity not in seen:
                seen.add(entity)
                seeds.append(entity)
        return seeds

    def _expand_graph(self, seeds: List[str]) -> List[Tuple[str, str, str]]:
        """Expand over the graph from seed entities."""

        if not seeds:
            return []

        if self.graph_mode == "path":
            triples = self.graph_client.expand_shortest_path(seeds, depth=self.path_depth)
        else:
            triples = self.graph_client.expand_k_hop(seeds, k=self.k)
        return list(triples)

    def _rerank_and_merge(
        self, results: Sequence[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Merge contexts while keeping citation metadata."""

        contexts: List[str] = []
        citations: List[Dict[str, Any]] = []
        seen_text = set()

        for result in results:
            text = result.get("text")
            if text and text not in seen_text:
                seen_text.add(text)
                contexts.append(text)
            citation = {key: result[key] for key in result if key not in {"text"}}
            if citation:
                citations.append(citation)

        return contexts, citations
