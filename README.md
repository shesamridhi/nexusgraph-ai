# ⬡ NexusGraph AI

> **Autonomous Multi-Agent Research Engine with Graph-Augmented Retrieval**  
> Built with LangGraph · Neo4j · ChromaDB · FastAPI · Streamlit

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.18-008CC1?style=flat-square&logo=neo4j)](https://neo4j.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🧠 What is NexusGraph AI?

NexusGraph AI is a production-grade **multi-agent research engine** that answers complex, multi-hop research questions by combining two retrieval strategies:

- **Vector Search** (ChromaDB) → handles *"What is X?"* — semantic similarity
- **Graph Traversal** (Neo4j) → handles *"How is A related to B?"* — relationship mapping
- **Hybrid** → combines both for 360° knowledge coverage

The secret is the **Think-Check-Execute loop** — a LangGraph state machine with three specialized agents:

```
User Question
     │
     ▼
┌─────────┐     ┌────────────┐     ┌────────┐
│  Router │────►│ Researcher │────►│ Critic │
│         │     │            │     │        │
│ Routes  │     │ Fetches    │     │ Scores │──── confidence < 0.7 ──► retry
│ intent  │     │ Vector+    │     │ answer │
│         │     │ Graph data │     │        │──── confidence ≥ 0.7 ──► Answer
└─────────┘     └────────────┘     └────────┘
```

---

## 🏗️ Architecture

```
nexusgraph/
├── backend/
│   ├── agents/
│   │   ├── graph_agent.py      # LangGraph: Router → Researcher → Critic
│   │   └── ingestion.py        # Entity extraction + graph + vector pipeline
│   ├── api/
│   │   └── main.py             # FastAPI application with all routes
│   ├── db/
│   │   ├── neo4j_client.py     # Neo4j async client + traversal queries
│   │   └── chroma_client.py    # ChromaDB client + embedding pipeline
│   ├── models/
│   │   └── schemas.py          # Pydantic v2 models (strict validation)
│   ├── utils/
│   │   └── seed_data.py        # Sample AI research papers (auto-seeded)
│   ├── tests/
│   │   └── test_nexusgraph.py  # Pytest test suite
│   └── config.py               # Settings via pydantic-settings
├── frontend/
│   └── app.py                  # Streamlit dashboard
├── docker-compose.yml          # Full stack: Neo4j + Chroma + API + UI
└── .env.example
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker + Docker Compose
- OpenAI API key ([get one here](https://platform.openai.com))

### 1. Clone & Configure

```bash
git clone https://github.com/shesamridhi/nexusgraph-ai.git
cd nexusgraph-ai

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Launch Everything

```bash
docker-compose up --build
```

That's it. Docker starts:
- **Neo4j** on `localhost:7474` (graph browser) + `7687` (bolt)
- **ChromaDB** on `localhost:8001`
- **FastAPI** on `localhost:8000` ([Swagger docs](http://localhost:8000/docs))
- **Streamlit** on `localhost:8501` ([Dashboard](http://localhost:8501))

> **First boot:** ~6 sample AI research papers are automatically seeded into both stores.

### 3. Ask Your First Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Find all papers by authors who collaborated with Ashish Vaswani", "max_hops": 2}'
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health + DB status |
| `POST` | `/query` | **Main:** Run the agent research loop |
| `POST` | `/ingest/text` | Ingest raw text into knowledge graph |
| `POST` | `/ingest/file` | Upload PDF or .txt file |
| `GET` | `/graph/stats` | Node/edge counts by type |
| `GET` | `/graph/related/{entity}` | Multi-hop graph traversal |
| `GET` | `/graph/coauthors/{name}` | Find co-authors |
| `GET` | `/vector/search?q=...` | Direct semantic search |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive Swagger UI |

### Example Query Request

```json
POST /query
{
  "question": "What methods did Google researchers use for language model pre-training?",
  "max_hops": 2,
  "stream": false
}
```

### Example Response

```json
{
  "query_id": "a3f9b2c1",
  "question": "What methods did Google researchers use...",
  "answer": "Google researchers introduced BERT...",
  "query_type": "hybrid",
  "confidence": 0.89,
  "sources": [...],
  "graph_context": {
    "nodes": [{"id": "paper_bert_002", "labels": ["Document"], ...}],
    "relationships": [{"source": "Jacob Devlin", "target": "paper_bert_002", "type": "WROTE"}]
  },
  "agent_trace": [
    {"step": 1, "agent": "Router", "action": "Classified as 'hybrid'..."},
    {"step": 2, "agent": "Researcher", "action": "Retrieved 5 vector chunks + 3 graph nodes"},
    {"step": 3, "agent": "Critic", "action": "Confidence: 0.89 | Verdict: pass"}
  ],
  "iterations": 1,
  "latency_ms": 1842.3
}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | LangGraph 0.1 | Agent state machine (Router → Researcher → Critic) |
| **LLM** | OpenAI GPT-4o-mini | Routing, answer generation, entity extraction |
| **Graph DB** | Neo4j 5.18 | Knowledge graph: nodes, edges, Cypher traversal |
| **Vector DB** | ChromaDB 0.5 | Semantic embeddings + similarity search |
| **Embeddings** | text-embedding-3-small | Document and query embeddings |
| **Backend** | FastAPI 0.111 | Async REST API |
| **Validation** | Pydantic v2 | Strict request/response schemas |
| **Monitoring** | LangSmith | Full agent trace observability |
| **Metrics** | Prometheus | API metrics via FastAPI Instrumentator |
| **Frontend** | Streamlit | Interactive research dashboard |
| **Container** | Docker Compose | Full-stack orchestration |
| **CI/CD** | GitHub Actions | Lint → Test → Build → Deploy |

---

## 🔭 LangSmith Monitoring

Set `LANGCHAIN_TRACING_V2=true` and add your `LANGCHAIN_API_KEY` in `.env`.  
Every agent step — Router decision, Researcher fetch, Critic evaluation — appears as a trace in your [LangSmith dashboard](https://smith.langchain.com).

---

## 🧪 Running Tests

```bash
cd backend
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## 🌐 Deployment

### Backend → Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Deploy from backend directory
cd backend
railway up
```

Set environment variables in Railway dashboard: `OPENAI_API_KEY`, `NEO4J_URI`, `LANGCHAIN_API_KEY`.

### Frontend → Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set `API_BASE_URL` to your Railway backend URL

---

## 📊 Sample Queries to Try

**Graph (relationship) queries:**
- `Find all papers written by authors who collaborated with Ashish Vaswani`
- `Which topics connect Transformers and RAG research?`

**Vector (semantic) queries:**
- `What is the self-attention mechanism and how does it work?`
- `Explain the difference between BERT and GPT architectures`

**Hybrid queries:**
- `What did Google Brain researchers contribute to NLP and what techniques did they use?`
- `Find papers about pre-training from researchers at Facebook AI Research`.

---

## 📄 License

MIT License — free to use, modify, and deploy.

---

<p align="center">
  Built by <a href="https://github.com/shesamridhi">@shesamridhi</a> · 
  <a href="http://localhost:8000/docs">API Docs</a> · 
  <a href="http://localhost:8501">Dashboard</a>
</p>
