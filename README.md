# RAG Pipeline (FastAPI + Chroma + Postgres)

This is a containerized Retrieval-Augmented Generation (RAG) service that lets you upload documents, indexes them into a vector database (Chroma), and answer questions grounded in your docs via an LLM (OpenAI or Gemini).

## Features
- Upload up to 20 documents per request (limits configurable)
- Supports PDF, DOCX, and TXT
- Chunking with token-aware overlap
- Vector database: ChromaDB (server) with persistence
- Metadata in Postgres
- REST API via FastAPI
- Pluggable providers:
  - Embeddings: OpenAI | Sentence-Transformers (local) | Fake (tests)
  - LLM: OpenAI | Gemini | Fake (tests)
- Dockerized with docker-compose
- Tests (unit + integration)

## Quickstart

1) Copy `.env.example` to `.env` and fill in values (OpenAI/Gemini keys if using those):cp .env.example .env

2) Build and run:docker compose up --build

3) API available at:
- http://localhost:8001/docs (Swagger UI)

## API Endpoints

- POST `/documents` (multipart/form-data)
  - Field: `files` (one or many)
  - Returns: processed metadata (id, file_name, pages, chunks)
- GET `/documents`
  - Returns: list of processed documents
- GET `/documents/{id}`
  - Returns: document metadata
- POST `/query`
  - Body: `{ "query": "text", "top_k": 5, "doc_ids": ["uuid", ...] }`
  - Returns: `{ answer, sources[], used_provider }`

## Configuration

- Vector store: `VECTOR_STORE=chroma` (default) or `memory` (for tests)
- Embeddings:
  - `EMBEDDING_PROVIDER=openai|local|fake`
  - `EMBEDDING_MODEL=text-embedding-3-small` (OpenAI)
  - `LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- LLM:
  - `LLM_PROVIDER=openai|gemini|fake`
  - `LLM_MODEL=gpt-4o-mini`
  - `GEMINI_MODEL=gemini-1.5-flash`

## Testing

Run tests locally (without Docker):pip install -r requirements.txt
pytest -q

Or in Docker (spin up api container and run a shell).

By default tests use `fake` LLM/embeddings and `memory` vector store for reproducibility and zero external deps.

## Deployment

### Local
- `docker compose up -d`
- Data persists in volumes `pgdata` and `chroma-data`

### Cloud options
- AWS ECS Fargate:
  - Push images to ECR
  - Create ECS service (api, chroma) + RDS Postgres
  - Set env vars/secrets in task definitions
  - Use an ALB pointing to api service
- GCP Cloud Run:
  - Run `api` and `chroma` as separate services (or use a managed vector DB)
  - Use Cloud SQL for Postgres
  - Connect services via VPC connectors
- Azure Container Apps:
  - Deploy `api` and `chroma`, provision Azure Database for PostgreSQL
  - Configure secrets and environment in ACA
- Simpler path: use a VM (EC2/Compute Engine) and run `docker compose up -d`
