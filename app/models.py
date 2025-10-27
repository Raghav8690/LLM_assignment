from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from .database import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String(512), nullable=False)
    content_type = Column(String(128), nullable=False)
    source_path = Column(String(1024), nullable=False)
    num_pages = Column(Integer, nullable=False, default=0)
    num_chunks = Column(Integer, nullable=False, default=0)
    status = Column(String(64), nullable=False, default="processed")  # processed | failed
    error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())