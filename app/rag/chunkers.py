from typing import List, Dict
import tiktoken

class TextChunker:
    def __init__(self, max_tokens: int = 800, overlap: int = 200, encoding_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.enc = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text or ""))

    def split(self, text: str) -> List[Dict]:
        tokens = self.enc.encode(text)
        chunks = []
        start = 0
        chunk_id = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            sub_tokens = tokens[start:end]
            chunk_text = self.enc.decode(sub_tokens)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": len(sub_tokens)
            })
            chunk_id += 1
            if end == len(tokens):
                break
            start = max(0, end - self.overlap)
        return chunks