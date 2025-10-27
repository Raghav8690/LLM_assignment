from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    env: str = Field(default="dev", alias="ENV")
    api_title: str = Field(default="RAG API", alias="API_TITLE")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8001, alias="API_PORT")

    db_url: str = Field(..., alias="DB_URL")

    vector_store: str = Field(default="chroma", alias="VECTOR_STORE")
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="rag_collection", alias="CHROMA_COLLECTION")

    max_docs_per_upload: int = Field(default=20, alias="MAX_DOCS_PER_UPLOAD")
    max_pages_per_doc: int = Field(default=1000, alias="MAX_PAGES_PER_DOC")
    chunk_tokens: int = Field(default=800, alias="CHUNK_TOKENS")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    top_k_default: int = Field(default=5, alias="TOP_K")

    embedding_provider: str = Field(default="openai", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    local_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="LOCAL_EMBEDDING_MODEL")

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    upload_dir: str = Field(default="/data/uploads", alias="UPLOAD_DIR")

    allowed_origins: List[str] = Field(default=["*"], alias="ALLOWED_ORIGINS")

    class Config:
        case_sensitive = False
        extra = "ignore"

settings = Settings()