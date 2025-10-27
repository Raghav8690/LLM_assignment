from app.rag.chunker import TextChunker

def test_chunker_splits_with_overlap():
    text = "hello world " * 1000
    c = TextChunker(max_tokens=50, overlap=10)
    chunks = c.split(text)
    assert len(chunks) > 1
    assert chunks[0]["token_count"] <= 50