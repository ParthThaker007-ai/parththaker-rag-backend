from fastapi import WebSocket, WebSocketDisconnect
import json
from app.services import language, embedding, vector_store, generator, hallucination_guard, cache
from app.config import config

# Import the same functions from routes or duplicate them here
from app.routes import expand_query, TOP_K, SCORE_THRESHOLD, RERANKER_AVAILABLE, rerank_hits

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
                
                detected_lang = language.detect_language(query)
                if not language.is_supported_language(detected_lang):
                    await manager.send_message(websocket, {
                        "type": "error",
                        "content": f"Unsupported language. Supported: {', '.join(config.SUPPORTED_LANGUAGES)}"
                    })
                    continue
                
                query_en = query
                if detected_lang != "en":
                    query_en = language.translate(query, detected_lang, "en")
                
                # Query expansion and search
                expanded_queries = expand_query(query_en)
                all_hits = []
                for q in expanded_queries:
                    try:
                        emb = embedding.get_embeddings([q])[0]
                    except Exception as e:
                        continue
                    filter_dict = {"source": paper_id} if paper_id and paper_id != "all" else {}
                    hits = vector_store.search(emb, top_k=TOP_K, score_threshold=SCORE_THRESHOLD, filter_dict=filter_dict)
                    all_hits.extend(hits)
                
                # Deduplicate
                seen = {}
                for h in all_hits:
                    if h["id"] not in seen or h["score"] > seen[h["id"]]["score"]:
                        seen[h["id"]] = h
                unique_hits = list(seen.values())
                
                if RERANKER_AVAILABLE:
                    unique_hits = rerank_hits(query_en, unique_hits, top_k=5)
                else:
                    unique_hits = sorted(unique_hits, key=lambda x: x["score"], reverse=True)[:5]
                
                if not unique_hits:
                    await manager.send_message(websocket, {
                        "type": "response",
                        "content": "I cannot find relevant information about this in the paper.",
                        "confidence": 0.0,
                        "citations": []
                    })
                    continue
                
                context = [h["payload"]["text"] for h in unique_hits]
                gen_result = generator.generate_response(query_en, context)
                response_en = gen_result["text"]
                
                fact_check = hallucination_guard.fact_check_response(response_en, unique_hits)
                confidence = fact_check["confidence"]
                
                citations = []
                for h in unique_hits[:5]:
                    citations.append({
                        "page": h["payload"].get("page"),
                        "source": h["payload"].get("source", "Unknown"),
                        "snippet": h["payload"]["text"][:200] + "..."
                    })
                
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