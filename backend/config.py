"""
NexusGraph AI — Configuration & Settings (Groq Edition)
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "NexusGraph AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Groq (Free LLM) ──────────────────────────────────
    GROQ_API_KEY: str = "gsk_placeholder"
    GROQ_MODEL: str = "llama3-70b-8192"

    # ── OpenAI (sirf placeholder, use nahi hoga) ─────────
    OPENAI_API_KEY: str = "placeholder"
    OPENAI_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Neo4j ────────────────────────────────────────────
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "nexusgraph123"

    # ── ChromaDB ─────────────────────────────────────────
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "nexusgraph_docs"

    # ── LangSmith ────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = "placeholder"
    LANGCHAIN_PROJECT: str = "nexusgraph-ai"

    # ── Retrieval ────────────────────────────────────────
    VECTOR_TOP_K: int = 5
    GRAPH_DEPTH: int = 2
    CRITIC_THRESHOLD: float = 0.7
    MAX_AGENT_ITERATIONS: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
