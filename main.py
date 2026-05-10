"""
NexusGraph AI — FastAPI Application
High-performance async API serving the multi-agent research engine.
"""
import structlog
import os
import uuid
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from config import get_settings
from models.schemas import (
    QueryRequest, QueryResponse, IngestRequest, IngestResponse,
    HealthResponse, GraphStatsResponse, ErrorResponse
)
from db.neo4j_client import neo4j_client
from db.chroma_client import chroma_client
from agents.graph_agent import run_research_query
from agents.ingestion import ingest_document, ingest_from_pdf_bytes

logger = structlog.get_logger(__name__)
settings = get_settings()


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to databases on startup, close on shutdown."""
    logger.info("nexusgraph.starting", version=settings.APP_VERSION)
    try:
        await neo4j_client.connect()
        logger.info("neo4j.ready")
    except Exception as e:
        logger.error("neo4j.connection_failed", error=str(e))

    try:
        await chroma_client.connect()
        logger.info("chroma.ready")
    except Exception as e:
        logger.error("chroma.connection_failed", error=str(e))

    # Seed sample data in background
    from utils.seed_data import seed_sample_data
    try:
        await seed_sample_data()
    except Exception as e:
        logger.warning("seed.failed", error=str(e))

    yield

    await neo4j_client.close()
    logger.info("nexusgraph.shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NexusGraph AI",
    description="Autonomous Multi-Agent Research Engine with Graph-Augmented Retrieval",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)


# ── Health & Status ───────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check():
    neo4j_status = await neo4j_client.health_check()
    chroma_status = await chroma_client.health_check()
    # Quick OpenAI check
    openai_status = "configured" if settings.OPENAI_API_KEY.startswith("sk-") and len(settings.OPENAI_API_KEY) > 10 else "not_configured"

    overall = "healthy" if all(
        s in ("healthy", "configured") for s in [neo4j_status, chroma_status, openai_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        neo4j=neo4j_status,
        chromadb=chroma_status,
        openai=openai_status,
    )


# ── Query Endpoint ────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Research"])
async def research_query(request: QueryRequest):
    """
    **The core endpoint.** Triggers the full Think-Check-Execute agent loop:
    1. Router classifies the question
    2. Researcher fetches from Vector DB + Graph DB
    3. Critic validates and optionally retries
    4. Returns a grounded answer with full agent trace
    """
    query_id = str(uuid.uuid4())[:8]
    logger.info("query.received", query_id=query_id, question=request.question[:80])

    response = await run_research_query(
        question=request.question,
        max_hops=request.max_hops,
        query_id=query_id,
    )
    return response


# ── Ingestion Endpoints ───────────────────────────────────────────────────────

@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestRequest):
    """
    Ingest raw text into the knowledge graph.
    Automatically extracts entities, builds Neo4j relationships, and chunks into ChromaDB.
    """
    result = await ingest_document(
        content=request.content,
        title=request.title,
        metadata=request.metadata,
    )
    return result


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a PDF or text file. The system will:
    - Extract text content
    - Run entity extraction (authors, topics, organizations)
    - Build vector embeddings + knowledge graph simultaneously
    """
    content = await file.read()
    filename = file.filename or "upload"

    if filename.lower().endswith(".pdf"):
        result = await ingest_from_pdf_bytes(
            pdf_bytes=content,
            filename=filename,
        )
    else:
        # Treat as plain text
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        result = await ingest_document(
            content=text,
            title=filename,
            metadata={"source": filename, "type": "text"},
        )
    return result


# ── Graph Endpoints ───────────────────────────────────────────────────────────

@app.get("/graph/stats", response_model=GraphStatsResponse, tags=["Graph"])
async def graph_stats():
    """Returns statistics about the knowledge graph (node/edge counts by type)."""
    stats = await neo4j_client.get_stats()
    return GraphStatsResponse(**stats)


@app.get("/graph/related/{entity}", tags=["Graph"])
async def get_related(
    entity: str,
    hops: int = Query(2, ge=1, le=4, description="Graph traversal depth")
):
    """Traverse the graph from a named entity and return related documents."""
    result = await neo4j_client.find_related_documents(entity_name=entity, max_hops=hops)
    return {
        "entity": entity,
        "hops": hops,
        "nodes_found": len(result.nodes),
        "relationships_found": len(result.relationships),
        "nodes": [{"id": n.id, "labels": n.labels, "properties": n.properties} for n in result.nodes],
        "relationships": [{"source": r.source, "target": r.target, "type": r.type} for r in result.relationships],
    }


@app.get("/graph/coauthors/{author_name}", tags=["Graph"])
async def get_coauthors(author_name: str):
    """Find all co-authors of a given author in the knowledge graph."""
    coauthors = await neo4j_client.find_co_authors(author_name)
    return {"author": author_name, "coauthors": coauthors, "count": len(coauthors)}


# ── Vector Endpoints ──────────────────────────────────────────────────────────

@app.get("/vector/search", tags=["Vector"])
async def vector_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
):
    """Direct semantic similarity search against the vector store."""
    results = await chroma_client.similarity_search(query=q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {"doc_id": r.doc_id, "score": r.score, "content": r.content[:300], "metadata": r.metadata}
            for r in results
        ]
    }


@app.get("/vector/stats", tags=["Vector"])
async def vector_stats():
    """Returns ChromaDB collection statistics."""
    return await chroma_client.get_collection_stats()


# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("unhandled_exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
