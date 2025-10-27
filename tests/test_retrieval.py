from app.rag.pipeline import RAGPipeline
from app.settings import settings

def test_pipeline_fake_providers(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    p = RAGPipeline(chunk_tokens=50, overlap=10)
    text = "This is a small test document about apples and bananas. Apples are red, bananas are yellow."
    doc_id = "00000000-0000-0000-0000-000000000001"
    num_chunks = p.index_document(doc_id, text, {"file_name": "test.txt"})
    assert num_chunks > 0
    answer, ctx = p.query("What color are bananas?", top_k=3)
    assert "(fake) Based on context" in answer
    assert len(ctx) > 0