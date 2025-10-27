from typing import List, Dict, Optional
from uuid import UUID, uuid4

from .chunker import TextChunker
from .embeddings import get_embeddings_provider
from .vector_store import get_vector_store, BaseVectorStore
from .llm import get_llm, SYSTEM_PROMPT
from .utils import clean_text

class RAGPipeline:
    def __init__(self, chunk_tokens: int, overlap: int):
        self.chunker = TextChunker(max_tokens=chunk_tokens, overlap=overlap)
        self.embedder = get_embeddings_provider()
        self.vs: BaseVectorStore = get_vector_store()
        self.llm = get_llm()

    def index_document(self, doc_id: UUID, text: str, base_meta: Dict) -> int:
        text = clean_text(text)
        chunks = self.chunker.split(text)
        if not chunks:
            return 0
        docs = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(docs)
        ids = [f"{doc_id}:{c['chunk_id']}" for c in chunks]
        metas = []
        for c in chunks:
            m = dict(base_meta)
            m.update({"doc_id": str(doc_id), "chunk_id": c["chunk_id"], "token_count": c["token_count"]})
            metas.append(m)
        self.vs.upsert(ids=ids, embeddings=embeddings, metadatas=metas, documents=docs)
        return len(chunks)

    def retrieve(self, query: str, top_k: int, doc_ids: Optional[List[str]] = None):
        q_emb = self.embedder.embed([query])[0]
        where = None
        if doc_ids:
            where = {"doc_id": {"$in": [str(d) for d in doc_ids]}}
        res = self.vs.query(embedding=q_emb, top_k=top_k, where=where)
        return res

    def answer(self, query: str, contexts: List[Dict]) -> str:
        # Build context with numbered sources
        numbered = []
        for i, c in enumerate(contexts, start=1):
            src = f"[{i}] {c.get('file_name')} (page {c.get('page', 'n/a')})"
            snippet = c.get("text", "")[:1200]
            numbered.append(f"{src}\n{snippet}\n")
        context_str = "\n---\n".join(numbered)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer concisely with citations."}
        ]
        return self.llm.generate(messages)

    def query(self, query: str, top_k: int = 5, doc_ids: Optional[List[str]] = None):
        res = self.retrieve(query, top_k, doc_ids)
        contexts = []
        for i, (doc, meta, dist) in enumerate(zip(res.documents, res.metadatas, res.distances)):
            contexts.append({
                "text": doc,
                "file_name": meta.get("file_name"),
                "page": meta.get("page"),
                "doc_id": meta.get("doc_id"),
                "chunk_id": meta.get("chunk_id"),
                "score": 1 - float(dist),
            })
        answer = self.answer(query, contexts)
        return answer, contexts