from fastapi import APIRouter, HTTPException
from app.models import QueryRequest, QueryResponse, HealthResponse, FeedbackRequest
from app.services import language, embedding, vector_store, generator, hallucination_guard, cache
from app.config import config
import time
import json
from datetime import datetime
import os

router = APIRouter()

# ============================================================
# QUERY EXPANSION – generate alternative phrasings
# ============================================================
def expand_query(query: str) -> list[str]:
    """Return a list of query variations to improve recall."""
    expansions = [query]
    # Simple rules – you can expand with more logic
    if "mCRS" in query:
        expansions.append("Mean Corruption Robustness Score")
        expansions.append("mCRS metric")
        expansions.append("mCRS formula")
        expansions.append("mCRS definition")
    if "FGTS" in query:
        expansions.append("Failure Geometry Transferability Score")
    # Add more as needed
    return expansions

# ============================================================
# RETRIEVAL PARAMETERS – tune for better recall
# ============================================================
TOP_K = 10
SCORE_THRESHOLD = 0.3

# ============================================================
# CROSS-ENCODER RERANKER (optional)
# ============================================================
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    reranker = None

def rerank_hits(query: str, hits: list, top_k: int = 5) -> list:
    """Rerank hits using a cross-encoder and return the top_k."""
    if not hits or not RERANKER_AVAILABLE:
        return hits[:top_k]
    pairs = [[query, h["payload"]["text"]] for h in hits]
    scores = reranker.predict(pairs)
    # Attach rerank scores and sort
    for i, h in enumerate(hits):
        h["rerank_score"] = float(scores[i])
    hits.sort(key=lambda x: x["rerank_score"], reverse=True)
    return hits[:top_k]

@router.get("/health", response_model=HealthResponse)
async def health_check():
    vector_db_ok = False
    try:
        info = vector_store.get_collection_info()
        vector_db_ok = info["points_count"] > 0
    except:
        vector_db_ok = False
    
    cache_ok = cache.get_redis_client() is not None
    groq_ok = bool(config.GROQ_API_KEY)
    
    return HealthResponse(
        status="ok",
        languages=config.SUPPORTED_LANGUAGES,
        vector_db=vector_db_ok,
        llm_available=groq_ok,
        cache_available=cache_ok
    )

@router.post("/query", response_model=QueryResponse)
async def query_paper(req: QueryRequest):
    start_time = time.time()
    
    # 1. Detect language
    detected_lang = language.detect_language(req.query)
    if not language.is_supported_language(detected_lang):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Supported: {', '.join(config.SUPPORTED_LANGUAGES)}"
        )
    
    # 2. Check cache
    cached = cache.get_cached_response(req.query, req.paper_id, detected_lang)
    if cached:
        return QueryResponse(**cached)
    
    # 3. Translate to English if needed
    query_en = req.query
    translated = False
    original_lang = detected_lang
    
    if detected_lang != "en":
        query_en = language.translate(req.query, detected_lang, "en")
        translated = True
    
    # 4. Query expansion
    expanded_queries = expand_query(query_en)
    
    # 5. Embed each expansion and search
    all_hits = []
    for q in expanded_queries:
        try:
            embeddings = embedding.get_embeddings([q])
            q_emb = embeddings[0]
        except Exception as e:
            continue
        
        filter_dict = {"source": req.paper_id} if req.paper_id else {}
        
        try:
            hits = vector_store.search(q_emb, top_k=TOP_K, score_threshold=SCORE_THRESHOLD, filter_dict=filter_dict)
            all_hits.extend(hits)
        except Exception as e:
            continue
    
    # Deduplicate by ID (keep highest score)
    seen = {}
    for h in all_hits:
        if h["id"] not in seen or h["score"] > seen[h["id"]]["score"]:
            seen[h["id"]] = h
    unique_hits = list(seen.values())
    
    # Rerank if available
    if RERANKER_AVAILABLE:
        unique_hits = rerank_hits(query_en, unique_hits, top_k=5)
    else:
        unique_hits = sorted(unique_hits, key=lambda x: x["score"], reverse=True)[:5]
    
    if not unique_hits:
        response_obj = QueryResponse(
            response="I cannot find relevant information about this in the paper. Please try rephrasing.",
            language=detected_lang,
            confidence=0.0,
            citations=[],
            sources=[],
            translated=translated,
            original_language=original_lang
        )
        cache.cache_response(req.query, response_obj.dict(), req.paper_id, detected_lang)
        return response_obj
    
    # 6. Generate response
    context = [h["payload"]["text"] for h in unique_hits]
    sources = unique_hits
    
    try:
        gen_result = generator.generate_response(query_en, context)
        response_en = gen_result["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    # 7. Fact-check (confidence)
    fact_check = hallucination_guard.fact_check_response(response_en, sources)
    confidence = fact_check["confidence"]
    warning = fact_check["warning"]
    
    # 8. Citations
    citations = []
    for h in unique_hits[:5]:
        citations.append({
            "page": h["payload"].get("page"),
            "source": h["payload"].get("source", "Unknown"),
            "snippet": h["payload"]["text"][:200] + ("..." if len(h["payload"]["text"]) > 200 else "")
        })
    
    # 9. Translate back if needed
    final_response = response_en
    if detected_lang != "en":
        try:
            final_response = language.translate(response_en, "en", detected_lang)
        except:
            pass
        if warning:
            final_response += f"\n\n{warning}"
    else:
        if warning:
            final_response += f"\n\n{warning}"
    
    response_obj = QueryResponse(
        response=final_response,
        language=detected_lang,
        confidence=confidence,
        citations=citations,
        sources=[{"page": h["payload"].get("page"), "score": h.get("rerank_score", h["score"])} for h in unique_hits[:5]],
        translated=translated,
        original_language=original_lang
    )
    
    cache.cache_response(req.query, response_obj.dict(), req.paper_id, detected_lang)
    
    elapsed = time.time() - start_time
    print(f"⏱️ Query: {elapsed:.3f}s | Lang: {detected_lang} | Conf: {confidence:.2f} | Hits: {len(unique_hits)}")
    
    return response_obj

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    os.makedirs("data", exist_ok=True)
    with open("data/feedback.json", "a") as f:
        f.write(json.dumps({
            "query_id": feedback.query_id,
            "helpful": feedback.helpful,
            "comment": feedback.comment,
            "timestamp": datetime.now().isoformat()
        }) + "\n")
    return {"status": "ok", "message": "Feedback recorded."}