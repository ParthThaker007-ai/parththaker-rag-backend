from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, 
    Distance, 
    VectorParams, 
    Filter, 
    FieldCondition, 
    MatchValue,
    PayloadSchemaType
)
from app.config import config
import uuid
import time

# timeout=60: the default qdrant-client timeout is quite short (a few
# seconds), which is fine for small reads/searches but too tight for
# upserting a batch of 50 points (each with a 384-dim vector + text
# payload) to a free-tier cloud cluster — especially under any network
# congestion. 60s gives batches enough headroom without hanging forever
# on a genuinely dead connection.
qdrant_client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
    timeout=60,
)

COLLECTION_NAME = config.QDRANT_COLLECTION

def ensure_collection(vector_size: int = 384):
    """Ensure collection exists with proper indexes."""
    collections = qdrant_client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    if COLLECTION_NAME not in collection_names:
        # Create collection
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"✅ Collection exists: {COLLECTION_NAME}")
    
    # Ensure payload index on 'source' field for filtering
    try:
        # Check if index exists
        # Qdrant doesn't have a direct "check index" endpoint; we'll create it (idempotent)
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source",
            field_type=PayloadSchemaType.KEYWORD,
        )
        print("✅ Created index on 'source' field")
    except Exception as e:
        # If index already exists, ignore the error (it may raise "already exists")
        if "already exists" not in str(e).lower():
            print(f"⚠️ Could not create index: {e}")

def upsert_chunks(chunks: list[dict], embeddings: list[list[float]], max_retries: int = 3):
    """
    Upload chunks + their embeddings to Qdrant, with retry-with-backoff
    on transient failures (timeouts, brief connection issues). A single
    write timeout shouldn't lose an entire batch of chunks during
    ingestion — that's the failure mode this was added to fix.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match")
    
    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        point_id = chunk.get("id") or str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=emb,
                payload={
                    "text": chunk["text"],
                    "source": chunk.get("source", "unknown"),
                    "page": chunk.get("page"),
                    "chunk_index": i,
                }
            )
        )
    
    last_error = None
    for attempt in range(max_retries):
        try:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"✅ Upserted {len(points)} chunks")
            return len(points)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 3 * (2 ** attempt)  # 3s, 6s, 12s
                print(f"⚠️ Upsert failed ({e}), retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

    # All retries exhausted — raise so ingest.py's existing try/except
    # around this call reports the batch as failed (as before), instead
    # of silently losing data.
    raise RuntimeError(f"Upsert failed after {max_retries} attempts: {last_error}")

def search(embedding: list[float], top_k: int = 5, score_threshold: float = 0.7, filter_dict: dict = None):
    search_filter = None
    if filter_dict:
        conditions = []
        for key, value in filter_dict.items():
            # If value is None or empty, skip filter
            if value:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if conditions:
            search_filter = Filter(must=conditions)
    
    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=search_filter,
    )
    
    results = []
    for hit in hits:
        results.append({
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        })
    
    return results

def get_collection_info():
    info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    return {
        "name": COLLECTION_NAME,
        "points_count": info.points_count,
        "status": info.status,
    }