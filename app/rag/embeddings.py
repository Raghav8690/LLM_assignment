from typing import List
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from ..settings import settings

class EmbeddingsProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class OpenAIEmbeddings(EmbeddingsProvider):
    def __init__(self, model: str, api_key: str | None):
        from openai import OpenAI
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

class LocalEmbeddings(EmbeddingsProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True).tolist()
        return vecs

class FakeEmbeddings(EmbeddingsProvider):
    # Deterministic pseudo-embeddings for tests
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.dim)
            v = v / np.linalg.norm(v)
            out.append(v.tolist())
        return out

def get_embeddings_provider() -> EmbeddingsProvider:
    prov = settings.embedding_provider.lower()
    if prov == "openai":
        return OpenAIEmbeddings(settings.embedding_model, settings.openai_api_key)
    if prov == "local":
        return LocalEmbeddings(settings.local_embedding_model)
    if prov == "fake":
        return FakeEmbeddings()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {prov}")