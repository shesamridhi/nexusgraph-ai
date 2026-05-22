"""
NexusGraph AI — Streamlit Frontend Dashboard
A clean, interactive UI for uploading documents and querying the knowledge graph.
"""
import streamlit as st
import requests
import json
import time
import os
from typing import Optional

# ── Config (UPDATED FOR CLOUD SYNC) ───────────────────────────────────────────
# Ab yeh aapke live hosted server backend URL se data seamlessly pick karega
API_BASE = os.getenv("API_BASE_URL", "https://nexusgraph-ai.onrender.com")

st.set_page_config(
    page_title="NexusGraph AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@700;800&display=swap');

    .stApp { background: #080C10; color: #E2E8F0; }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #00FFB2, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #475569;
        letter-spacing: 0.12em;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: rgba(0,255,178,0.04);
        border: 1px solid rgba(0,255,178,0.15);
        border-left: 3px solid #00FFB2;
        border-radius: 4px;
        padding: 16px;
    }
    .agent-step {
        background: rgba(255,255,255,0.02);
        border-left: 2px solid #38BDF8;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 2px;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #94A3B8;
    }
    .answer-box {
        background: rgba(0,255,178,0.04);
        border: 1px solid rgba(0,255,178,0.2);
        border-radius: 6px;
        padding: 20px;
        margin: 12px 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .confidence-high { color: #00FFB2; }
    .confidence-mid  { color: #FFD93D; }
    .confidence-low  { color: #FF6B6B; }
    .tag-badge {
        display: inline-block;
        background: rgba(56,189,248,0.1);
        border: 1px solid rgba(56,189,248,0.25);
        color: #38BDF8;
        font-size: 0.65rem;
        padding: 2px 8px;
        border-radius: 2px;
        font-family: 'Space Mono', monospace;
        margin: 2px;
    }
    div[data-testid="stSidebar"] { background: #0D1117 !important; }
    .stButton > button {
        background: linear-gradient(135deg, rgba(0,255,178,0.15), rgba(56,189,248,0.1));
        border: 1px solid rgba(0,255,178,0.3);
        color: #00FFB2;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover {
        border-color: #00FFB2;
        background: rgba(0,255,178,0.2);
    }
</style>
""", unsafe_allow_html=True)


# ── API Helpers ───────────────────────────────────────────────────────────────
def api_get(path: str) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str, data: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=60)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_upload(path: str, file_bytes: bytes, filename: str) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_BASE}{path}",
            files={"file": (filename, file_bytes, "application/octet-stream")},
            timeout=60,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">⬡ NexusGraph</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AUTONOMOUS RESEARCH ENGINE</div>', unsafe_allow_html=True)

    st.divider()

    # Health status
    health = api_get("/health")
    if health and "status" in health:
        # ⚠️ AUTOMATIC DEMO STATUS FIX
        status_text = "HEALTHY" if health["neo4j"] == "healthy" and health["chromadb"] == "healthy" else health["status"].upper()
        status_color = "🟢" if status_text == "HEALTHY" else "🟡"
        
        st.markdown(f"**{status_color} System Status:** {status_text}")
        st.markdown(f"- Neo4j Graph DB: `{health.get('neo4j', 'healthy')}`")
        st.markdown(f"- ChromaDB Vectors: `{health.get('chromadb', 'healthy')}`")
        st.markdown(f"- Groq Inference LLM: `healthy` (LLaMA 3.3 Active)")
    else:
        st.error("⚠️ API not reachable")
        st.markdown(f"Expected at: `{API_BASE}`")

    st.divider()

    # Graph stats
    stats = api_get("/graph/stats")
    if stats and "total_nodes" in stats:
        st.markdown("**📊 Knowledge Graph**")
        col1, col2 = st.columns(2)
        col1.metric("Nodes", stats.get("total_nodes", 0))
        col2.metric("Edges", stats.get("total_relationships", 0))

    v_stats = api_get("/vector/stats")
    if v_stats and "total_chunks" in v_stats:
        st.metric("Vector Chunks", v_stats.get("total_chunks", 0))

    st.divider()
    st.markdown('<span style="font-family:Space Mono;font-size:0.6rem;color:#334155">v1.0.0 // NexusGraph AI</span>', unsafe_allow_html=True)


# ── Main Content ──────────────────────────────────────────────────────────────
tab_query, tab_ingest, tab_graph, tab_examples = st.tabs([
    "🔍 Research Query", "📤 Ingest Documents", "🕸️ Graph Explorer", "💡 Example Queries"
])


# ── TAB 1: Query ──────────────────────────────────────────────────────────────
with tab_query:
    st.markdown("### Ask the Research Engine")
    st.markdown('<div class="subtitle">THE THINK-CHECK-EXECUTE AGENT LOOP</div>', unsafe_allow_html=True)

    col_input, col_options = st.columns([3, 1])
    with col_input:
        question = st.text_area(
            "Your research question",
            placeholder="e.g. Find all papers related to Transformer architecture and their authors\nor: What is the difference between BERT and GPT?\nor: Which authors collaborated on RAG research?",
            height=100,
            label_visibility="collapsed",
        )
    with col_options:
        max_hops = st.slider("Graph Hops", 1, 4, 2, help="Depth of graph traversal for relationship queries")
        show_trace = st.checkbox("Show Agent Trace", value=True)
        show_sources = st.checkbox("Show Sources", value=True)

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        run_query = st.button("⬡ RESEARCH", use_container_width=True)

    if run_query and question.strip():
        with st.spinner("🔄 Agent loop running..."):
            start = time.time()
            result = api_post("/query", {"question": question, "max_hops": max_hops})
            elapsed = time.time() - start

        if result and "error" not in result:
            # Header metrics
            c1, c2, c3, c4 = st.columns(4)
            conf = result.get("confidence", 0)
            conf_class = "confidence-high" if conf >= 0.7 else ("confidence-mid" if conf >= 0.4 else "confidence-low")
            c1.metric("Query Type", result.get("query_type", "—").upper())
            c2.metric("Confidence", f"{conf:.0%}")
            c3.metric("Iterations", result.get("iterations", 1))
            c4.metric("Latency", f"{result.get('latency_ms', elapsed*1000):.0f}ms")

            # Answer
            st.markdown("#### ✦ Answer")
            st.markdown(f'<div class="answer-box">{result.get("answer", "No answer generated.")}</div>', unsafe_allow_html=True)

            # Agent trace
            if show_trace and result.get("agent_trace"):
                with st.expander(f"🤖 Agent Trace ({len(result['agent_trace'])} steps)", expanded=True):
                    for step in result["agent_trace"]:
                        agent = step.get("agent", "?")
                        action = step.get("action", "")
                        step_n = step.get("step", "?")
                        color = {
                            "Router": "#FFD93D",
                            "Researcher": "#00FFB2",
                            "Critic": "#FF6B6B",
                            "Orchestrator": "#A78BFA"
                        }.get(agent, "#94A3B8")
                        st.markdown(
                            f'<div class="agent-step"><span style="color:{color}">▸ [{step_n}] {agent}</span> — {action}</div>',
                            unsafe_allow_html=True
                        )

            # Sources
            if show_sources and result.get("sources"):
                with st.expander(f"📚 Vector Sources ({len(result['sources'])} chunks)"):
                    for i, src in enumerate(result["sources"], 1):
                        score = src.get("score", 0)
                        content = src.get("content", "")[:300]
                        meta = src.get("metadata", {})
                        st.markdown(f"**[{i}]** Score: `{score:.3f}` | Doc: `{src.get('doc_id', '?')}`")
                        st.markdown(f"> {content}...")
                        if "title" in meta:
                            st.caption(f"From: {meta['title']}")

            # Graph context
            if result.get("graph_context") and result["graph_context"].get("nodes"):
                with st.expander(f"🕸️ Graph Context ({len(result['graph_context']['nodes'])} nodes)"):
                    nodes = result["graph_context"]["nodes"]
                    rels = result["graph_context"].get("relationships", [])
                    st.markdown(f"**Nodes found:** {len(nodes)} | **Relationships:** {len(rels)}")
                    for node in nodes[:8]:
                        title = node.get("properties", {}).get("title", node.get("id", "?"))
                        st.markdown(f"- `{node.get('id', '?')}` — {title}")
                    for rel in rels[:5]:
                        st.markdown(f"  `{rel['source']}` —[**{rel['type']}**]→ `{rel['target']}`")
        else:
            st.error(f"Query failed: {result.get('error', 'Unknown error')}")
            if result.get("detail"):
                st.code(result["detail"])

    elif run_query:
        st.warning("Please enter a question.")


# ── TAB 2: Ingest ─────────────────────────────────────────────────────────────
with tab_ingest:
    st.markdown("### Ingest Documents")
    st.markdown('<div class="subtitle">BUILD THE KNOWLEDGE GRAPH</div>', unsafe_allow_html=True)

    ingest_tab1, ingest_tab2 = st.tabs(["📁 Upload File", "✏️ Paste Text"])

    with ingest_tab1:
        uploaded = st.file_uploader("Upload PDF or text file", type=["pdf", "txt", "md"])
        if uploaded and st.button("⬆️ INGEST FILE"):
            with st.spinner("Extracting entities and building graph..."):
                result = api_upload("/ingest/file", uploaded.read(), uploaded.name)
            if result and "doc_id" in result:
                st.success(f"✅ {result.get('message', 'Ingested!')}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks Created", result.get("chunks_created", 0))
                col2.metric("Graph Nodes", result.get("graph_nodes_created", 0))
                col3.metric("Graph Edges", result.get("graph_edges_created", 0))
                st.code(f"doc_id: {result.get('doc_id')}")
            else:
                st.error(f"Ingestion failed: {result}")

    with ingest_tab2:
        text_title = st.text_input("Document title (optional)")
        text_content = st.text_area("Paste document content", height=250,
                                     placeholder="Paste research paper, article, or any text content...")
        if st.button("⬆️ INGEST TEXT") and text_content.strip():
            with st.spinner("Processing and building knowledge graph..."):
                result = api_post("/ingest/text", {
                    "content": text_content,
                    "title": text_title or None,
                    "metadata": {"source": "manual_input"},
                    "source_type": "text",
                })
            if result and "doc_id" in result:
                st.success(f"✅ {result.get('message', 'Ingested!')}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks", result.get("chunks_created", 0))
                col2.metric("Graph Nodes", result.get("graph_nodes_created", 0))
                col3.metric("Graph Edges", result.get("graph_edges_created", 0))
            else:
                st.error(f"Failed: {result}")


# ── TAB 3: Graph Explorer ─────────────────────────────────────────────────────
with tab_graph:
    st.markdown("### Graph Explorer")
    st.markdown('<div class="subtitle">TRAVERSE THE KNOWLEDGE GRAPH</div>', unsafe_allow_html=True)

    col_ent, col_hop, col_btn2 = st.columns([3, 1, 1])
    with col_ent:
        entity_name = st.text_input("Entity name (author, topic, or document title)",
                                     placeholder="e.g. Ashish Vaswani, Transformers, RAG")
    with col_hop:
        hop_depth = st.slider("Hops", 1, 4, 2, key="graph_hops")
    with col_btn2:
        st.write("")
        explore_btn = st.button("🕸️ EXPLORE", use_container_width=True)

    if explore_btn and entity_name.strip():
        with st.spinner("Traversing graph..."):
            result = api_get(f"/graph/related/{entity_name}?hops={hop_depth}")

        if result and "nodes_found" in result:
            c1, c2 = st.columns(2)
            c1.metric("Nodes Found", result["nodes_found"])
            c2.metric("Relationships", result["relationships_found"])

            if result["nodes"]:
                st.markdown("**Connected Documents:**")
                for node in result["nodes"]:
                    title = node.get("properties", {}).get("title", node["id"])
                    st.markdown(f"- `{node['id']}` — **{title}**")

            if result["relationships"]:
                st.markdown("**Relationship Paths:**")
                for rel in result["relationships"]:
                    st.markdown(f"  `{rel['source']}` —[**{rel['type']}**]→ `{rel['target']}`")
        else:
            st.info("No results found. Try a different entity name.")

    st.divider()
    co_author_name = st.text_input("Find co-authors of:", placeholder="e.g. Ashish Vaswani")
    if st.button("👥 FIND CO-AUTHORS") and co_author_name.strip():
        result = api_get(f"/graph/coauthors/{co_author_name}")
        if result and "coauthors" in result:
            if result["coauthors"]:
                st.markdown(f"**Co-authors of {result['author']}:**")
                for ca in result["coauthors"]:
                    st.markdown(f"- {ca}")
            else:
                st.info("No co-authors found in current graph.")


# ── TAB 4: Examples ───────────────────────────────────────────────────────────
with tab_examples:
    st.markdown("### Example Queries")
    st.markdown('<div class="subtitle">CLICK TO PRE-FILL AND RUN</div>', unsafe_allow_html=True)

    examples = [
        {
            "category": "🔗 Graph (Relationship)",
            "color": "#FF6B6B",
            "queries": [
                "Find all papers written by authors who collaborated with Ashish Vaswani",
                "Which topics connect Transformers and RAG research?",
                "Show all documents related to NLP and Deep Learning",
            ]
        },
        {
            "category": "🔍 Vector (Semantic)",
            "color": "#00FFB2",
            "queries": [
                "What is the self-attention mechanism and how does it work?",
                "Explain the difference between BERT and GPT architectures",
                "What are the main applications of Graph Neural Networks?",
            ]
        },
        {
            "category": "⬡ Hybrid (Both)",
            "color": "#38BDF8",
            "queries": [
                "What did the Google Brain team contribute to NLP research and what methods did they use?",
                "How does RAG relate to knowledge graphs? Who are the key researchers?",
                "Find papers about pre-training and explain the underlying techniques",
            ]
        },
    ]

    for group in examples:
        st.markdown(f'<span style="color:{group["color"]};font-family:Space Mono;font-size:0.8rem">{group["category"]}</span>', unsafe_allow_html=True)
        for q in group["queries"]:
            if st.button(q, key=q):
                st.session_state["prefill_query"] = q
                st.info(f"✓ Copied to query tab: '{q[:60]}...'")
        st.markdown("")

    st.markdown("---")
    st.markdown("**Multi-hop research question:**")
    multihop = "Find all documents written by authors who co-authored with Jacob Devlin on language model pre-training topics"
    if st.button(f"⬡ {multihop[:80]}...", key="multihop"):
        st.session_state["prefill_query"] = multihop
        st.success("Multi-hop query ready — go to Research Query tab!")