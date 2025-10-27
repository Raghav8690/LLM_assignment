from fastapi.testclient import TestClient
from app.main import app
from app.settings import settings
import io

def test_upload_and_query(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("VECTOR_STORE", "memory")

    client = TestClient(app)
    content = b"Bananas are yellow.\nApples are red."
    files = {"files": ("colors.txt", io.BytesIO(content), "text/plain")}
    r = client.post("/documents", files=files)
    assert r.status_code == 200
    docs = r.json()
    assert len(docs) == 1

    r2 = client.post("/query", json={"query": "What color are bananas?", "top_k": 2})
    assert r2.status_code == 200
    body = r2.json()
    assert "answer" in body
    assert len(body["sources"]) >= 1