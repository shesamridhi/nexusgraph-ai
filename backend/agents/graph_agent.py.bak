"""
NexusGraph AI — The Agentic Brain (LangGraph + Groq)
Think-Check-Execute loop:
  Router → Researcher → Critic → (retry if needed) → Answer
"""
import structlog
import json
from typing import TypedDict, Annotated, Optional, List
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import operator
import uuid

from config import get_settings
from models.schemas import (
    QueryType, AgentStatus, AgentStep,
    VectorResult, GraphResult, QueryResponse
)
from db.neo4j_client import neo4j_client
from db.chroma_client import chroma_client

logger = structlog.get_logger(__name__)
settings = get_settings()


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query_id: str
    question: str
    max_hops: int
    query_type: QueryType
    extracted_entities: List[str]
    extracted_keywords: List[str]
    vector_results: List[VectorResult]
    graph_result: Optional[GraphResult]
    combined_context: str
    confidence: float
    critique: str
    needs_retry: bool
    retry_count: int
    answer: str
    agent_trace: Annotated[List[AgentStep], operator.add]
    status: AgentStatus
    error: Optional[str]


# ── LLM Factory (Groq) ────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=temperature,
        groq_api_key=settings.GROQ_API_KEY,
        max_retries=3,
    )


# ── Agent 1: Router ───────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are a query router for a research knowledge graph system.
Analyze the user's question and extract:
1. query_type: one of "vector" (semantic/definition questions), "graph" (relationship/multi-hop questions), or "hybrid" (both needed)
2. entities: named entities like person names, organization names, document titles
3. keywords: key concepts and topics for semantic search

Rules:
- Use "graph" when the question asks about relationships, collaborations, connections, "who worked with", "all X related to Y"
- Use "vector" for "what is", "explain", "describe", "summarize" questions
- Use "hybrid" when both context and relationships are needed

Respond ONLY with valid JSON, no markdown, no extra text:
{"query_type": "...", "entities": [...], "keywords": [...]}"""


async def router_node(state: AgentState) -> dict:
    step_num = len(state["agent_trace"]) + 1
    logger.info("agent.router", query=state["question"][:80])

    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Question: {state['question']}")
    ]

    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        query_type = QueryType(parsed.get("query_type", "hybrid"))
        entities = parsed.get("entities", [])
        keywords = parsed.get("keywords", [])
    except Exception as e:
        logger.warning("router.parse_failed", error=str(e))
        query_type = QueryType.HYBRID
        entities = []
        keywords = state["question"].split()[:5]

    trace_step = AgentStep(
        step=step_num, agent="Router",
        action=f"Classified as '{query_type.value}' | Entities: {entities} | Keywords: {keywords}"
    )
    return {
        "query_type": query_type,
        "extracted_entities": entities,
        "extracted_keywords": keywords,
        "status": AgentStatus.ROUTING,
        "agent_trace": [trace_step],
    }


# ── Agent 2: Researcher ────────────────────────────────────────────────────────

async def researcher_node(state: AgentState) -> dict:
    step_num = len(state["agent_trace"]) + 1
    logger.info("agent.researcher", type=state["query_type"], retry=state.get("retry_count", 0))

    vector_results: List[VectorResult] = []
    graph_result: Optional[GraphResult] = None
    context_parts = []

    # Vector retrieval
    if state["query_type"] in (QueryType.VECTOR, QueryType.HYBRID):
        try:
            top_k = settings.VECTOR_TOP_K + (state.get("retry_count", 0) * 3)
            vector_results = await chroma_client.similarity_search(
                query=state["question"], top_k=min(top_k, 15)
            )
            if vector_results:
                context_parts.append("=== SEMANTIC SEARCH RESULTS ===")
                for i, r in enumerate(vector_results[:5], 1):
                    context_parts.append(f"[{i}] (score={r.score:.3f}) {r.content[:400]}")
        except Exception as e:
            logger.error("researcher.vector_error", error=str(e))

    # Graph retrieval
    if state["query_type"] in (QueryType.GRAPH, QueryType.HYBRID):
        try:
            if state["extracted_entities"]:
                primary_entity = state["extracted_entities"][0]
                graph_result = await neo4j_client.find_related_documents(
                    entity_name=primary_entity,
                    max_hops=state.get("max_hops", 2)
                )
                if graph_result.nodes:
                    context_parts.append("\n=== GRAPH RELATIONSHIP RESULTS ===")
                    context_parts.append(f"Found {len(graph_result.nodes)} related documents:")
                    for node in graph_result.nodes[:8]:
                        title = node.properties.get("title", node.id)
                        context_parts.append(f"  • {title}")
                    if graph_result.relationships:
                        for rel in graph_result.relationships[:5]:
                            context_parts.append(f"  {rel.source} --[{rel.type}]--> {rel.target}")

            if state["extracted_keywords"]:
                kw_graph = await neo4j_client.keyword_subgraph(state["extracted_keywords"])
                if kw_graph.nodes:
                    context_parts.append("\n=== TOPIC-BASED GRAPH RESULTS ===")
                    for node in kw_graph.nodes[:5]:
                        context_parts.append(f"  • {node.properties.get('title', node.id)}")
                    if graph_result is None:
                        graph_result = kw_graph
        except Exception as e:
            logger.error("researcher.graph_error", error=str(e))

    combined_context = "\n".join(context_parts) if context_parts else "No relevant context found in the knowledge base."

    trace_step = AgentStep(
        step=step_num, agent="Researcher",
        action=f"Retrieved {len(vector_results)} vector chunks + {len(graph_result.nodes) if graph_result else 0} graph nodes",
        result=f"Context length: {len(combined_context)} chars"
    )
    return {
        "vector_results": vector_results,
        "graph_result": graph_result,
        "combined_context": combined_context,
        "status": AgentStatus.RESEARCHING,
        "agent_trace": [trace_step],
    }


# ── Agent 3: Critic ────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are a rigorous fact-checker for an AI research assistant.
Given a question, retrieved context, and a draft answer, evaluate:
1. Does the answer directly address the question?
2. Is the answer grounded in the provided context (no hallucination)?
3. Is the context sufficient, or should we retrieve more?

Respond ONLY with valid JSON, no markdown, no extra text:
{
  "confidence": 0.0-1.0,
  "verdict": "pass" or "retry",
  "critique": "brief explanation",
  "missing_info": "what additional info would help (if retry)"
}"""

ANSWER_SYSTEM = """You are NexusGraph AI, an expert research assistant with access to a hybrid knowledge graph.
Generate a comprehensive, well-structured answer based ONLY on the provided context.
If the context is insufficient, say so clearly. Cite sources when possible.
Format your answer in clear paragraphs. Be precise and avoid speculation."""


async def critic_node(state: AgentState) -> dict:
    step_num = len(state["agent_trace"]) + 1
    logger.info("agent.critic", retry_count=state.get("retry_count", 0))

    llm = get_llm(temperature=0.1)

    # Generate draft answer
    answer_messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

Context:
{state['combined_context']}

Generate a comprehensive answer:""")
    ]
    answer_response = await llm.ainvoke(answer_messages)
    draft_answer = answer_response.content.strip()

    # Critique it
    critique_messages = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

Retrieved Context:
{state['combined_context'][:2000]}

Draft Answer:
{draft_answer}

Evaluate this answer:""")
    ]
    try:
        critique_response = await llm.ainvoke(critique_messages)
        raw = critique_response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        evaluation = json.loads(raw)
        confidence = float(evaluation.get("confidence", 0.5))
        verdict = evaluation.get("verdict", "pass")
        critique_text = evaluation.get("critique", "")
    except Exception as e:
        logger.warning("critic.parse_failed", error=str(e))
        confidence = 0.6
        verdict = "pass"
        critique_text = "Evaluation parsing failed; proceeding with draft."

    retry_count = state.get("retry_count", 0)
    needs_retry = (
        verdict == "retry"
        and confidence < settings.CRITIC_THRESHOLD
        and retry_count < settings.MAX_AGENT_ITERATIONS - 1
    )

    trace_step = AgentStep(
        step=step_num, agent="Critic",
        action=f"Confidence: {confidence:.2f} | Verdict: {verdict} | Retry: {needs_retry}",
        result=critique_text[:200]
    )
    return {
        "answer": draft_answer,
        "confidence": confidence,
        "critique": critique_text,
        "needs_retry": needs_retry,
        "retry_count": retry_count + (1 if needs_retry else 0),
        "status": AgentStatus.CRITIQUING,
        "agent_trace": [trace_step],
    }


# ── Conditional Edge ──────────────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    if state.get("needs_retry", False):
        logger.info("agent.retry", count=state["retry_count"])
        return "researcher"
    return "end"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"researcher": "researcher", "end": END},
    )
    return graph.compile()


# ── Public Entry Point ────────────────────────────────────────────────────────

async def run_research_query(question: str, max_hops: int = 2,
                              query_id: str = None) -> QueryResponse:
    import time
    if query_id is None:
        query_id = str(uuid.uuid4())[:8]

    start_time = time.time()
    graph = build_agent_graph()

    initial_state: AgentState = {
        "query_id": query_id,
        "question": question,
        "max_hops": max_hops,
        "query_type": QueryType.UNKNOWN,
        "extracted_entities": [],
        "extracted_keywords": [],
        "vector_results": [],
        "graph_result": None,
        "combined_context": "",
        "confidence": 0.0,
        "critique": "",
        "needs_retry": False,
        "retry_count": 0,
        "answer": "",
        "agent_trace": [],
        "status": AgentStatus.ROUTING,
        "error": None,
    }

    try:
        final_state = await graph.ainvoke(initial_state)
        latency = (time.time() - start_time) * 1000

        final_trace = final_state["agent_trace"]
        final_trace.append(AgentStep(
            step=len(final_trace) + 1,
            agent="Orchestrator",
            action=f"Complete | {len(final_trace)} steps | {latency:.0f}ms",
        ))

        return QueryResponse(
            query_id=query_id,
            question=question,
            answer=final_state["answer"],
            query_type=final_state["query_type"],
            confidence=final_state["confidence"],
            sources=final_state["vector_results"][:5],
            graph_context=final_state["graph_result"],
            agent_trace=final_trace,
            iterations=final_state["retry_count"] + 1,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error("agent.failed", error=str(e), query_id=query_id)
        return QueryResponse(
            query_id=query_id,
            question=question,
            answer=f"Agent pipeline failed: {str(e)}. Please check your configuration.",
            query_type=QueryType.UNKNOWN,
            confidence=0.0,
            agent_trace=[AgentStep(step=1, agent="Orchestrator", action=f"Error: {str(e)}")],
            iterations=1,
            latency_ms=round(latency, 2),
        )
