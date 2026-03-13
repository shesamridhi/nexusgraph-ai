"""
NexusGraph AI — Pydantic Models
All request/response schemas with strict validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class AgentStatus(str, Enum):
    ROUTING = "routing"
    RESEARCHING = "researching"
    CRITIQUING = "critiquing"
    COMPLETE = "complete"
    FAILED = "failed"


# ── Request Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="The research question")
    collection: Optional[str] = Field(None, description="Specific collection to query")
    max_hops: int = Field(2, ge=1, le=4, description="Max graph traversal hops")
    stream: bool = Field(False, description="Stream agent steps as SSE")

    model_config = {"json_schema_extra": {"example": {
        "question": "Find all papers by authors who collaborated with Dr. Smith on NLP topics",
        "max_hops": 2,
        "stream": False
    }}}


class IngestRequest(BaseModel):
    source_type: Literal["text", "url"] = "text"
    content: str = Field(..., min_length=10)
    title: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


# ── Response Models ──────────────────────────────────────────────────────────

class VectorResult(BaseModel):
    doc_id: str
    content: str
    score: float
    metadata: dict = {}


class GraphNode(BaseModel):
    id: str
    labels: List[str]
    properties: dict


class GraphRelationship(BaseModel):
    source: str
    target: str
    type: str
    properties: dict = {}


class GraphResult(BaseModel):
    nodes: List[GraphNode] = []
    relationships: List[GraphRelationship] = []
    paths: List[dict] = []


class AgentStep(BaseModel):
    step: int
    agent: str
    action: str
    result: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    query_type: QueryType
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[VectorResult] = []
    graph_context: Optional[GraphResult] = None
    agent_trace: List[AgentStep] = []
    iterations: int = 1
    latency_ms: float
    langsmith_run_url: Optional[str] = None

    model_config = {"json_schema_extra": {"example": {
        "query_id": "abc-123",
        "question": "What is RAG?",
        "answer": "RAG (Retrieval-Augmented Generation) is...",
        "query_type": "vector",
        "confidence": 0.92,
        "latency_ms": 1240.5
    }}}


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    graph_nodes_created: int
    graph_edges_created: int
    collection: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    neo4j: str
    chromadb: str
    openai: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GraphStatsResponse(BaseModel):
    total_nodes: int
    total_relationships: int
    node_labels: dict
    relationship_types: dict


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    query_id: Optional[str] = None
