from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID

class DocumentCreateResponse(BaseModel):
    id: UUID
    file_name: str
    content_type: str
    num_pages: int
    num_chunks: int
    status: str

class DocumentMetadata(BaseModel):
    id: UUID
    file_name: str
    content_type: str
    num_pages: int
    num_chunks: int
    status: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    doc_ids: Optional[List[UUID]] = None

class SourceChunk(BaseModel):
    doc_id: UUID
    file_name: str
    page: int | None = None
    chunk_id: int
    score: float
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    used_provider: str