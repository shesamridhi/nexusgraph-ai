import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from config import get_settings
settings = get_settings()

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    content: str
    metadata: dict


class ChromaClient:
    def __init__(self):
        self._client = None
        self._collection = None
        self._embeddings = None

    async def connect(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_sync)

    def _connect_sync(self):
        try:
            # Embedded mode — no separate ChromaDB server needed
            self._client = chromadb.Client()  # 0.4.24 compatible (EphemeralClient = Client)
            logger.info("chroma.connected mode=embedded")
        except Exception as e:
            logger.error(f"chroma.failed: {e}")
            raise

        # Use DefaultEmbeddingFunction from 0.4.24 — lightweight, no large ONNX download
        self._embeddings = embedding_functions.DefaultEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=self._embeddings,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"chroma.collection_ready name={settings.CHROMA_COLLECTION}")

    async def disconnect(self):
        self._client = None
        self._collection = None

    def _check(self):
        if self._collection is None:
            raise RuntimeError("ChromaDB not initialized.")

    async def health_check(self) -> str:
        try:
            self._check()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._collection.count)
            return "healthy"
        except Exception as e:
            return f"unhealthy: {e}"

    async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        self._check()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._collection.upsert(
            documents=documents, metadatas=metadatas, ids=ids))
        return True

    async def ingest_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        return await self.add_documents(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
        )

    async def similarity_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        self._check()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self._collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"],
        ))
        output = []
        if results and results.get("documents"):
            for doc, meta, dist, id_ in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                output.append(SearchResult(
                    doc_id=id_,
                    score=round(1 - dist, 4),
                    content=doc,
                    metadata=meta,
                ))
        return output

    async def query(self, query_text: str, n_results: int = 5, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        results = await self.similarity_search(query=query_text, top_k=n_results)
        return [{"document": r.content, "metadata": r.metadata, "score": r.score} for r in results]

    async def get_collection_stats(self) -> Dict[str, Any]:
        self._check()
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, self._collection.count)
        return {
            "collection_name": settings.CHROMA_COLLECTION,
            "document_count": count,
            "status": "connected",
        }

    async def delete_documents(self, ids: List[str]) -> bool:
        self._check()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._collection.delete(ids=ids))
        return True

    async def clear_collection(self) -> bool:
        self._check()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.delete_collection(settings.CHROMA_COLLECTION))
        self._collection = await loop.run_in_executor(None, lambda: self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=self._embeddings,
            metadata={"hnsw:space": "cosine"},
        ))
        return True


chroma_client = ChromaClient()