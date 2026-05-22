"""
NexusGraph AI — Neo4j Graph Database Layer
Handles all graph operations: node creation, relationship mapping, traversal.
"""
import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from config import get_settings
from models.schemas import GraphNode, GraphRelationship, GraphResult

logger = structlog.get_logger(__name__)
settings = get_settings()


class Neo4jClient:
    """Async Neo4j client with connection pooling."""

    def __init__(self):
        self._driver: Optional[AsyncDriver] = None

    async def connect(self):
        try:
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                max_connection_pool_size=50,
                connection_timeout=30,
                max_transaction_retry_time=30,
            )
            await self._driver.verify_connectivity()
            await self._create_constraints()
            logger.info("neo4j.connected", uri=settings.NEO4J_URI)
        except Exception as e:
            logger.error("neo4j.connection_failed", error=str(e))
            self._driver = None  # Important: None set karo failure pe

    async def close(self):
        if self._driver:
            await self._driver.close()

    async def _create_constraints(self):
        """Idempotent schema constraints."""
        if self._driver is None:
            return
        constraints = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
        ]
        async with self._driver.session() as session:
            for cypher in constraints:
                try:
                    await session.run(cypher)
                except Exception:
                    pass  # Already exists

    # ── Write Operations ─────────────────────────────────────────────────────

    async def upsert_document(self, doc_id: str, title: str, content_preview: str,
                               metadata: Dict[str, Any]) -> bool:
        if self._driver is None:
            return False
        cypher = """
        MERGE (d:Document {id: $doc_id})
        SET d.title = $title,
            d.preview = $preview,
            d.source = $source,
            d.created_at = datetime()
        RETURN d.id as id
        """
        async with self._driver.session() as session:
            result = await session.run(cypher,
                doc_id=doc_id, title=title,
                preview=content_preview[:300],
                source=metadata.get("source", "upload")
            )
            return bool(await result.single())

    async def upsert_author(self, name: str, affiliation: str = "") -> bool:
        if self._driver is None:
            return False
        cypher = """
        MERGE (a:Author {name: $name})
        SET a.affiliation = $affiliation,
            a.updated_at = datetime()
        RETURN a.name
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, name=name, affiliation=affiliation)
            return bool(await result.single())

    async def upsert_topic(self, name: str, category: str = "general") -> bool:
        if self._driver is None:
            return False
        cypher = """
        MERGE (t:Topic {name: $name})
        SET t.category = $category,
            t.updated_at = datetime()
        RETURN t.name
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, name=name, category=category)
            return bool(await result.single())

    async def create_relationship(self, from_id: str, from_label: str,
                                   to_id: str, to_label: str,
                                   rel_type: str, properties: Dict = None) -> bool:
        if self._driver is None:
            return False
        cypher = f"""
        MATCH (a:{from_label} {{{'name' if from_label != 'Document' else 'id'}: $from_id}})
        MATCH (b:{to_label} {{{'name' if to_label != 'Document' else 'id'}: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.weight = coalesce(r.weight, 0) + 1,
            r.updated_at = datetime()
        RETURN type(r) as rel_type
        """
        async with self._driver.session() as session:
            try:
                result = await session.run(cypher, from_id=from_id, to_id=to_id)
                return bool(await result.single())
            except Exception as e:
                logger.warning("neo4j.rel_failed", error=str(e))
                return False

    # ── Read / Traversal Operations ──────────────────────────────────────────

    async def find_related_documents(self, entity_name: str,
                                      max_hops: int = 2) -> GraphResult:
        """Multi-hop graph traversal — the core 'relationship query'."""
        if self._driver is None:
            return GraphResult()
        cypher = """
        MATCH path = (start)-[*1..$hops]-(d:Document)
        WHERE (start.name = $name OR start.id = $name)
        WITH d, path, length(path) as hops
        ORDER BY hops
        LIMIT 20
        RETURN
            collect(DISTINCT {id: d.id, title: d.title, preview: d.preview}) as docs,
            collect(DISTINCT {
                source: startNode(relationships(path)[0]).name,
                target: endNode(relationships(path)[0]).name,
                type: type(relationships(path)[0])
            }) as rels
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, name=entity_name, hops=max_hops)
            record = await result.single()
            if not record:
                return GraphResult()

            nodes = [
                GraphNode(id=d["id"], labels=["Document"],
                          properties={"title": d.get("title", ""), "preview": d.get("preview", "")})
                for d in (record["docs"] or []) if d.get("id")
            ]
            relationships = [
                GraphRelationship(source=r["source"] or "", target=r["target"] or "", type=r["type"] or "RELATED")
                for r in (record["rels"] or []) if r.get("source") and r.get("target")
            ]
            return GraphResult(nodes=nodes, relationships=relationships)

    async def find_co_authors(self, author_name: str) -> List[str]:
        """Find all co-authors of a given author."""
        if self._driver is None:
            return []
        cypher = """
        MATCH (a:Author {name: $name})-[:WROTE]->(d:Document)<-[:WROTE]-(coauthor:Author)
        WHERE coauthor.name <> $name
        RETURN DISTINCT coauthor.name as name
        LIMIT 10
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, name=author_name)
            records = await result.data()
            return [r["name"] for r in records]

    async def keyword_subgraph(self, keywords: List[str]) -> GraphResult:
        """Find subgraph matching any of the given keywords/topics."""
        if self._driver is None:
            return GraphResult()
        cypher = """
        UNWIND $keywords as kw
        MATCH (t:Topic)-[:COVERS]->(d:Document)
        WHERE toLower(t.name) CONTAINS toLower(kw)
        WITH d, collect(t.name) as topics
        RETURN
            collect(DISTINCT {id: d.id, title: d.title, topics: topics}) as docs
        LIMIT 15
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, keywords=keywords)
            record = await result.single()
            if not record:
                return GraphResult()
            nodes = [
                GraphNode(id=d["id"], labels=["Document"],
                          properties={"title": d.get("title", ""), "topics": d.get("topics", [])})
                for d in (record["docs"] or []) if d.get("id")
            ]
            return GraphResult(nodes=nodes)

    async def get_stats(self) -> Dict[str, Any]:
        """Graph database statistics."""
        if self._driver is None:
            return {
                "total_nodes": 0,
                "total_relationships": 0,
                "node_labels": {},
                "relationship_types": {},
            }
        async with self._driver.session() as session:
            node_result = await session.run(
                "MATCH (n) RETURN labels(n)[0] as label, count(*) as count"
            )
            rel_result = await session.run(
                "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count"
            )
            nodes = {r["label"]: r["count"] for r in await node_result.data() if r["label"]}
            rels = {r["type"]: r["count"] for r in await rel_result.data() if r["type"]}
            return {
                "total_nodes": sum(nodes.values()),
                "total_relationships": sum(rels.values()),
                "node_labels": nodes,
                "relationship_types": rels,
            }

    async def health_check(self) -> str:
        if self._driver is None:
            return "unavailable"
        try:
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            return "healthy"
        except ServiceUnavailable:
            return "unavailable"
        except Exception as e:
            return f"error: {str(e)}"


# Singleton
neo4j_client = Neo4jClient()