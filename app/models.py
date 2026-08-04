
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    paper_id: Optional[str] = None
    language: Optional[str] = "en"

class Citation(BaseModel):
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str
    source: Optional[str] = None

class Source(BaseModel):
    page: Optional[int] = None
    score: float
    text: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    language: str
    confidence: float
    citations: List[Citation] = []
    sources: List[Source] = []
    translated: bool = False
    original_language: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    languages: List[str]
    vector_db: bool
    llm_available: bool
    cache_available: bool

class FeedbackRequest(BaseModel):
    query_id: str
    helpful: bool
    comment: Optional[str] = None