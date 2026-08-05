from fastapi import WebSocket, WebSocketDisconnect
import json
from app.services import language, embedding, vector_store, generator, hallucination_guard, cache
from app.config import config

# Import shared helper functions from routes (if you want to avoid duplication)
# We'll copy the needed functions here to keep websocket self-contained.

# Query expansion (optional)
def expand_query(query: str) -> list[str]:
    expansions = [query]
    if "mCRS" in query:
        expansions.append("Mean Corruption Robustness Score")
        expansions.append("mCRS metric")
        expansions.append("mCRS formula")
        expansions.append("mCRS definition")
    if "FGTS" in query:
        expansions.append("Failure Geometry Transferability Score")
    return expansions

TOP_K = 10
SCORE_THRESHOLD = 0.3

# Cross-encoder reranker (optional)
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    reranker = None

def rerank_hits(query: str, hits: list, top_k: int = 5) -> list:
    if not hits or not RERANKER_AVAILABLE:
        return hits[:top_k]
    pairs = [[query, h["payload"]["text"]] for h in hits]
    scores = reranker.predict(pairs)
    for i, h in enumerate(hits):
        h["rerank_score"] = float(scores[i])
    hits.sort(key=lambda x: x["rerank_score"], reverse=True)
    return hits[:top_k]


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message: dict):
        await websocket.send_text(json.dumps(message))


manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket, paper_id: str = None):
    """
    WebSocket handler for chat.
    paper_id can be 'all', a specific paper ID, or None.
    """
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                query = msg.get("query", "")
                language_pref = msg.get("language", "en")

                if not query:
                    continue

                # 1. Detect language
                detected_lang = language.detect_language(query)
                if not language.is_supported_language(detected_lang):
                    await manager.send_message(websocket, {
                        "type": "error",
                        "content": f"Unsupported language. Supported: {', '.join(config.SUPPORTED_LANGUAGES)}"
                    })
                    continue

                # 2. Translate to English if needed
                query_en = query
                if detected_lang != "en":
                    query_en = language.translate(query, detected_lang, "en")

                # 3. Query expansion and search
                expanded_queries = expand_query(query_en)
                all_hits = []

                for q in expanded_queries:
                    try:
                        emb = embedding.get_embeddings([q])[0]
                    except Exception:
                        continue

                    # ✅ FIX: Only filter by source if paper_id is set and not "all"
                    filter_dict = {}
                    if paper_id and paper_id != "all":
                        filter_dict["source"] = paper_id

                    try:
                        hits = vector_store.search(
                            emb,
                            top_k=TOP_K,
                            score_threshold=SCORE_THRESHOLD,
                            filter_dict=filter_dict
                        )
                        all_hits.extend(hits)
                    except Exception:
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

                # 4. No hits -> fallback
                if not unique_hits:
                    await manager.send_message(websocket, {
                        "type": "response",
                        "content": "I cannot find relevant information about this in the paper. Please try rephrasing.",
                        "confidence": 0.0,
                        "citations": []
                    })
                    continue

                # 5. Generate response
                context = [h["payload"]["text"] for h in unique_hits]
                gen_result = generator.generate_response(query_en, context)
                response_en = gen_result["text"]

                # 6. Fact-check (confidence)
                fact_check = hallucination_guard.fact_check_response(response_en, unique_hits)
                confidence = fact_check["confidence"]

                # 7. Citations
                citations = []
                for h in unique_hits[:5]:
                    citations.append({
                        "page": h["payload"].get("page"),
                        "source": h["payload"].get("source", "Unknown"),
                        "snippet": h["payload"]["text"][:200] + "..."
                    })

                # 8. Translate back if needed
                final_response = response_en
                if detected_lang != "en":
                    final_response = language.translate(response_en, "en", detected_lang)

                await manager.send_message(websocket, {
                    "type": "response",
                    "content": final_response,
                    "confidence": confidence,
                    "citations": citations,
                    "lang": detected_lang
                })

            except json.JSONDecodeError:
                await manager.send_message(websocket, {"type": "error", "content": "Invalid JSON"})
            except Exception as e:
                await manager.send_message(websocket, {"type": "error", "content": str(e)})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("🔌 WebSocket disconnected")
