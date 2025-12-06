"""Load graph data into Neo4j from Parquet inputs."""

import argparse
import logging
from typing import Dict, Iterable, Tuple

import pandas as pd
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _create_constraints(tx) -> None:
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)")


def _merge_node(tx, node_data: Dict) -> None:
    node_id = node_data.get("id")
    if node_id is None:
        raise ValueError("Node data must contain an 'id' field")

    label = node_data.get("label") or "Entity"
    properties = {k: v for k, v in node_data.items() if k not in {"id", "label"}}

    cypher = f"MERGE (n:Entity {{id: $id}}) SET n:{label} SET n += $props"
    tx.run(cypher, id=node_id, props=properties)


def _merge_edge(tx, edge_data: Dict) -> None:
    source_id = edge_data.get("source_id") or edge_data.get("source")
    target_id = edge_data.get("target_id") or edge_data.get("target")
    if source_id is None or target_id is None:
        raise ValueError("Edge data must contain 'source_id'/'source' and 'target_id'/'target' fields")

    rel_type = edge_data.get("type") or "RELATED"
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


def _iter_records(df: pd.DataFrame) -> Iterable[Dict]:
    return df.to_dict(orient="records")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load graph data into Neo4j.")
    parser.add_argument("--nodes", required=True, help="Path to nodes parquet file")
    parser.add_argument("--edges", required=True, help="Path to edges parquet file")
    parser.add_argument("--uri", default="neo4j://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j user")
    parser.add_argument("--password", required=True, help="Neo4j password")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("Loading nodes from %s", args.nodes)
    logger.info("Loading edges from %s", args.edges)

    nodes_df = pd.read_parquet(args.nodes)
    edges_df = pd.read_parquet(args.edges)

    node_count, rel_count = load_to_neo4j(
        _iter_records(nodes_df), _iter_records(edges_df), args.uri, (args.user, args.password)
    )
    logger.info("Neo4j graph now has %s nodes and %s edges", node_count, rel_count)


if __name__ == "__main__":
    main()
