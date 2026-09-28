import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

import rag

MAX_UPLOAD = 10 * 1024 * 1024


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY")
    if expected and not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(401, "invalid or missing X-API-Key")


auth = [Depends(require_key)]
app = FastAPI(title="Hybrid RAG")


class Doc(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2_000_000)
    source: str | None = None


class IngestReq(BaseModel):
    documents: list[Doc] = Field(min_length=1, max_length=100)


class QueryReq(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    alpha: float = Field(0.5, ge=0, le=1)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest", dependencies=auth)
def ingest(req: IngestReq):
    return {"chunks": {d.id: rag.ingest(d.id, d.text, d.source or d.id) for d in req.documents}}


@app.post("/ingest/file", dependencies=auth)
def ingest_file(file: UploadFile):
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (10MB max)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(415, "only UTF-8 text files (.txt, .md) supported")
    name = file.filename or "upload"
    return {"chunks": {name: rag.ingest(name, text, name)}}


@app.post("/query", dependencies=auth)
def query(req: QueryReq):
    return rag.answer(req.question, req.top_k, req.alpha)
