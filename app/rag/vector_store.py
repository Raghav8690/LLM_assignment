from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

from ..settings import settings

@dataclass
class SearchResult:
    ids: List[str]
    metadatas: List[Dict[str, Any]]
    documents: List[str]
    distances: List[float]

class BaseVectorStore:
    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str]):
        raise NotImplementedError

    def query(self, embedding: List[float], top_k: int, where: Optional[Dict] = None) -> SearchResult:
        raise NotImplementedError

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self.client = chromadb.Client(ChromaSettings(
            chroma_api_impl="rest",
            chroma_server_host=settings.chroma_host,
            chroma_server_http_port=settings.chroma_port
        ))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str]):
        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, embedding: List[float], top_k: int, where: Optional[Dict] = None) -> SearchResult:
        res = self.collection.query(query_embeddings=[embedding], n_results=top_k, where=where or {})
        return SearchResult(
            ids=res.get("ids", [[]])[0],
            metadatas=res.get("metadatas", [[]])[0],
            documents=res.get("documents", [[]])[0],
            distances=res.get("distances", [[]])[0] or res.get("distances", [[]])[0]
        )

class InMemoryVectorStore(BaseVectorStore):
    # Simple cosine search for tests
    def __init__(self):
        self._embeds: List[np.ndarray] = []
        self._ids: List[str] = []
        self._metas: List[Dict] = []
        self._docs: List[str] = []

    def upsert(self, ids, embeddings, metadatas, documents):
        for i, e, m, d in zip(ids, embeddings, metadatas, documents):
            self._ids.append(i)
            self._embeds.append(np.array(e, dtype=np.float32))
            self._metas.append(m)
            self._docs.append(d)

    def query(self, embedding, top_k, where=None) -> SearchResult:
        emb = np.array(embedding, dtype=np.float32)
        # cosine similarity
        sims = []
        for idx, e in enumerate(self._embeds):
            if where:
                ok = True
                for k, v in where.items():
                    if isinstance(v, dict) and "$in" in v:
                        if self._metas[idx].get(k) not in v["$in"]:
                            ok = False
                            break
                    else:
                        if self._metas[idx].get(k) != v:
                            ok = False
                            break
                if not ok:
                    continue
            sim = float(np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e) + 1e-9))
            sims.append((sim, idx))
        sims.sort(reverse=True, key=lambda x: x[0])
        sims = sims[:top_k]
        return SearchResult(
            ids=[self._ids[i] for _, i in sims],
            metadatas=[self._metas[i] for _, i in sims],
            documents=[self._docs[i] for _, i in sims],
            distances=[1 - s for s, _ in sims],
        )

def get_vector_store() -> BaseVectorStore:
    if settings.vector_store.lower() == "chroma":
        return ChromaVectorStore(settings.chroma_collection)
    if settings.vector_store.lower() == "memory":
        return InMemoryVectorStore()
    raise ValueError(f"Unsupported VECTOR_STORE: {settings.vector_store}")