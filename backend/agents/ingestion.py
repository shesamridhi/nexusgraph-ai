"""
NexusGraph AI — Ingestion Pipeline (Groq Edition)
Parses documents → extracts entities → builds knowledge graph + vector store.
"""
import structlog
import json
import uuid
import re
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from db.neo4j_client import neo4j_client
from db.chroma_client import chroma_client
from models.schemas import IngestResponse

logger = structlog.get_logger(__name__)
settings = get_settings()

ENTITY_EXTRACTION_SYSTEM = """You are a knowledge extraction engine.
Given a document, extract:
1. authors: list of author names (people who wrote/created this)
2. topics: list of main topics/concepts (3-8 topics)
3. organizations: list of organizations mentioned
4. title: the document title (or derive one)

Respond ONLY with valid JSON, no markdown, no extra text:
{
  "title": "...",
  "authors": ["name1", "name2"],
  "topics": ["topic1", "topic2"],
  "organizations": ["org1", "org2"]
}"""


async def extract_entities(content: str, title: str = None) -> Dict[str, Any]:
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY,
    )
    preview = content[:2000]
    messages = [
        SystemMessage(content=ENTITY_EXTRACTION_SYSTEM),
        HumanMessage(content=f"Document:\n{preview}")
    ]
    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        if title:
            parsed["title"] = title
        return parsed
    except Exception as e:
        logger.warning("entity_extraction.failed", error=str(e))
        return {
            "title": title or "Untitled Document",
            "authors": [],
            "topics": _simple_keyword_extract(content),
            "organizations": [],
        }


def _simple_keyword_extract(text: str) -> list:
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:6]


async def ingest_document(content: str, title: str = None,
                           metadata: Dict[str, Any] = None,
                           doc_id: str = None) -> IngestResponse:
    if metadata is None:
        metadata = {}
    if doc_id is None:
        doc_id = str(uuid.uuid4())[:12]

    logger.info("ingestion.start", doc_id=doc_id, title=title, length=len(content))

    entities = await extract_entities(content, title)
    doc_title = entities.get("title", title or "Untitled")
    authors = entities.get("authors", [])
    topics = entities.get("topics", [])

    full_metadata = {
        **metadata,
        "title": doc_title,
        "authors": json.dumps(authors),
        "topics": json.dumps(topics),
    }

    chunks_created = await chroma_client.ingest_document(
        doc_id=doc_id, content=content, metadata=full_metadata,
    )

    graph_nodes = 0
    graph_edges = 0

    await neo4j_client.upsert_document(
        doc_id=doc_id, title=doc_title,
        content_preview=content[:300], metadata=metadata,
    )
    graph_nodes += 1

    for author in authors:
        if author and len(author) > 2:
            await neo4j_client.upsert_author(name=author)
            created = await neo4j_client.create_relationship(
                from_id=author, from_label="Author",
                to_id=doc_id, to_label="Document",
                rel_type="WROTE",
            )
            graph_nodes += 1
            if created:
                graph_edges += 1

    if len(authors) > 1:
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                await neo4j_client.create_relationship(
                    from_id=authors[i], from_label="Author",
                    to_id=authors[j], to_label="Author",
                    rel_type="CO_AUTHORED_WITH",
                )
                graph_edges += 1

    for topic in topics:
        if topic and len(topic) > 2:
            await neo4j_client.upsert_topic(name=topic)
            created = await neo4j_client.create_relationship(
                from_id=topic, from_label="Topic",
                to_id=doc_id, to_label="Document",
                rel_type="COVERS",
            )
            graph_nodes += 1
            if created:
                graph_edges += 1

    logger.info("ingestion.complete", doc_id=doc_id, chunks=chunks_created,
                nodes=graph_nodes, edges=graph_edges)

    return IngestResponse(
        doc_id=doc_id,
        chunks_created=chunks_created,
        graph_nodes_created=graph_nodes,
        graph_edges_created=graph_edges,
        collection=settings.CHROMA_COLLECTION,
        message=f"Successfully ingested '{doc_title}' with {len(authors)} authors and {len(topics)} topics.",
    )


async def ingest_from_pdf_bytes(pdf_bytes: bytes, filename: str,
                                 metadata: Dict[str, Any] = None) -> IngestResponse:
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text.strip())
        content = "\n\n".join(text_parts)
        if not content.strip():
            content = f"[PDF: {filename} - text extraction returned empty]"
    except Exception as e:
        content = f"[PDF parsing error for {filename}: {str(e)}]"

    return await ingest_document(
        content=content,
        title=filename.replace(".pdf", "").replace("_", " "),
        metadata={**(metadata or {}), "source": filename, "type": "pdf"},
    )
