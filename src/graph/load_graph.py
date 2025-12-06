"""Load topology CSV outputs into Neo4j."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import pandas as pd
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_TOPOLOGY_DIR = Path("data/topology")


def _create_constraints(tx) -> None:
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.label)")


def _sanitize_label(raw: str | None) -> str:
    if not raw:
        return "Entity"
    safe = re.sub(r"[^A-Za-z0-9_]", "", raw)
    return safe or "Entity"


def _merge_node(tx, node_data: Dict) -> None:
    node_id = node_data.get("id")
    if not node_id:
        raise ValueError("Node data must contain an 'id' field")

    class_label = _sanitize_label(node_data.get("class_label"))
    properties = {k: v for k, v in node_data.items() if k not in {"id", "class_label"}}

    cypher = f"MERGE (n:Entity {{id: $id}}) SET n:{class_label} SET n += $props"
    tx.run(cypher, id=node_id, props=properties)


def _merge_edge(tx, edge_data: Dict) -> None:
    source_id = edge_data.get("source_id") or edge_data.get("source")
    target_id = edge_data.get("target_id") or edge_data.get("target")
    if source_id is None or target_id is None:
        raise ValueError("Edge data must contain 'source_id'/'source' and 'target_id'/'target' fields")

    rel_type = _sanitize_label(edge_data.get("type"))
    properties = {k: v for k, v in edge_data.items() if k not in {"source_id", "source", "target_id", "target", "type"}}

    cypher = (
        "MERGE (s:Entity {id: $source_id}) "
        "MERGE (t:Entity {id: $target_id}) "
        f"MERGE (s)-[r:{rel_type}]->(t) "
        "SET r += $props"
    )
    tx.run(cypher, source_id=source_id, target_id=target_id, props=properties)


def _count_graph(tx) -> Tuple[int, int]:
    node_count = tx.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
    rel_count = tx.run("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS count").single()["count"]
    return node_count, rel_count


def load_to_neo4j(nodes: Iterable[Dict], edges: Iterable[Dict], uri: str, auth: Tuple[str, str]) -> Tuple[int, int]:
    """Load node and edge data into Neo4j and return imported counts."""

    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        with driver.session() as session:
            session.execute_write(_create_constraints)

            for node in nodes:
                session.execute_write(_merge_node, node)

            for edge in edges:
                session.execute_write(_merge_edge, edge)

            node_count, rel_count = session.execute_read(_count_graph)
            logger.info("Imported %s nodes and %s edges", node_count, rel_count)
            return node_count, rel_count
    finally:
        driver.close()


def _iter_records(df: pd.DataFrame) -> Iterator[Dict]:
    return df.to_dict(orient="records")


def _load_nodes(nodes_path: Path, provenance_path: Path | None) -> pd.DataFrame:
    frames = []
    if nodes_path.exists():
        frames.append(pd.read_csv(nodes_path))
    if provenance_path and provenance_path.exists():
        frames.append(pd.read_csv(provenance_path))
    if not frames:
        raise FileNotFoundError("No node CSVs found to import")

    nodes_df = pd.concat(frames, ignore_index=True).fillna("")
    nodes_df = nodes_df.rename(columns={":ID": "id", ":LABEL": "class_label"})
    return nodes_df


def _load_edges(edges_path: Path) -> pd.DataFrame:
    if not edges_path.exists():
        raise FileNotFoundError(f"Edge CSV not found: {edges_path}")
    edges_df = pd.read_csv(edges_path).fillna("")
    edges_df = edges_df.rename(
        columns={":START_ID": "source_id", ":END_ID": "target_id", ":TYPE": "type"}
    )
    return edges_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load topology CSVs into Neo4j.")
    parser.add_argument(
        "--nodes",
        default=str(DEFAULT_TOPOLOGY_DIR / "nodes.csv"),
        help="Path to nodes CSV exported by topology_detection.py",
    )
    parser.add_argument(
        "--provenance",
        default=str(DEFAULT_TOPOLOGY_DIR / "provenance.csv"),
        help="Optional provenance CSV to import as nodes",
    )
    parser.add_argument(
        "--edges",
        default=str(DEFAULT_TOPOLOGY_DIR / "rels.csv"),
        help="Path to relationships CSV exported by topology_detection.py",
    )
    parser.add_argument("--uri", default="neo4j://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j user")
    parser.add_argument("--password", required=True, help="Neo4j password")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    nodes_path = Path(args.nodes)
    provenance_path = Path(args.provenance) if args.provenance else None
    edges_path = Path(args.edges)

    logger.info("Loading nodes from %s", nodes_path)
    if provenance_path:
        logger.info("Loading provenance from %s", provenance_path)
    logger.info("Loading edges from %s", edges_path)

    nodes_df = _load_nodes(nodes_path, provenance_path)
    edges_df = _load_edges(edges_path)

    node_count, rel_count = load_to_neo4j(
        _iter_records(nodes_df), _iter_records(edges_df), args.uri, (args.user, args.password)
    )
    logger.info("Neo4j graph now has %s nodes and %s edges", node_count, rel_count)


if __name__ == "__main__":
    main()
