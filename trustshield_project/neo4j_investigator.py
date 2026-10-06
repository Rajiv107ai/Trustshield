"""Neo4j Graph Investigation Layer for TrustShield.

Provides multi-hop relationship exploration and Cypher query capabilities for human fraud analysts:
- Query: Find buyers connected to a seller through shared hardware devices within a time window
- Query: Detect closed collusion transaction cycles (Buyer -> Seller -> Shared Address/Device -> Buyer)
- Export: Serializes suspicious candidate rings into executable Cypher scripts for Neo4j Bloom visualization
- Standalone execution: Features an in-memory graph query simulator so queries are fully executable and testable without requiring a running Neo4j daemon.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import networkx as nx


class Neo4jInvestigator:
    """Investigator query layer providing Cypher query generation and in-memory execution."""

    def __init__(self, in_memory_graph: Optional[nx.MultiGraph] = None):
        self.graph = in_memory_graph or nx.MultiGraph()

    @staticmethod
    def cypher_shared_device_query(seller_id: str, days: int = 30) -> str:
        """Generate Cypher query to find buyers connected to a seller through a shared device."""
        return f"""
        MATCH (s:Seller {{seller_id: '{seller_id}'}})-[r1:USES_DEVICE]->(d:Device)<-[r2:USES_DEVICE]-(b:Buyer)
        WHERE r1.first_seen_date >= date() - duration({{days: {days}}})
        RETURN b.buyer_id AS buyer_id, d.device_id AS shared_device_id, b.orders_before AS orders_before
        ORDER BY b.orders_before DESC;
        """.strip()

    @staticmethod
    def cypher_collusion_cycle_query(min_txns: int = 3) -> str:
        """Generate Cypher query to detect triangular merchant-buyer device/address collusion."""
        return f"""
        MATCH path = (b:Buyer)-[t:TRANSACTS_WITH]->(s:Seller)-[:USES_ADDRESS]->(a:Address)<-[:USES_ADDRESS]-(b)
        WITH b, s, a, count(t) AS txn_count
        WHERE txn_count >= {min_txns}
        RETURN b.buyer_id AS buyer_id, s.seller_id AS seller_id, a.address_id AS shared_address_id, txn_count
        ORDER BY txn_count DESC;
        """.strip()

    def query_buyers_connected_via_device_in_memory(self, seller_id: str) -> List[Dict[str, Any]]:
        """Simulate the multi-hop Cypher query on the in-memory graph."""
        if not self.graph.has_node(seller_id):
            return []

        # Find devices used by seller
        seller_devices = set()
        for neighbor in self.graph.neighbors(seller_id):
            for edge_data in self.graph.get_edge_data(seller_id, neighbor, default={}).values():
                if edge_data.get("relation") == "uses_device" or "device" in str(neighbor).lower():
                    seller_devices.add(neighbor)

        results = []
        for dev in seller_devices:
            for neighbor in self.graph.neighbors(dev):
                if neighbor == seller_id:
                    continue
                for edge_data in self.graph.get_edge_data(dev, neighbor, default={}).values():
                    if edge_data.get("relation") == "uses_device" or "buyer" in str(neighbor).lower():
                        results.append({
                            "buyer_id": neighbor,
                            "shared_device_id": dev,
                            "connection_type": "hardware_collision",
                        })
        return results

    @staticmethod
    def export_cluster_to_cypher(cluster_nodes: List[str], edges: List[Dict[str, Any]]) -> str:
        """Generate Cypher CREATE statements to import a candidate fraud ring into Neo4j."""
        statements = ["// TrustShield Candidate Ring Cypher Export", "BEGIN;"]
        for node in cluster_nodes:
            lbl = "Buyer" if "B" in node else "Seller"
            statements.append(f"MERGE (n:{lbl} {{id: '{node}'}});")

        for e in edges:
            u, v, rel = e["src"], e["dst"], e.get("relation", "LINKED_TO")
            statements.append(
                f"MATCH (a {{id: '{u}'}}), (b {{id: '{v}'}}) MERGE (a)-[:{rel.upper()}]->(b);"
            )
        statements.append("COMMIT;")
        return "\n".join(statements)
