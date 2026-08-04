from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.config import config
from app.routes import router
from app.websocket import handle_websocket
from app.services import vector_store, cache

app = FastAPI(
    title="Parth Thaker — Paper RAG System",
    description="Multi-lingual RAG system (EN/DE/FR) with Groq + Upstash",
    version="2.0.0"
)

# CORS – allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST routes
app.include_router(router, prefix="/api")

# WebSocket endpoint – accept all connections
@app.websocket("/ws/{paper_id}")
async def websocket_endpoint(websocket: WebSocket, paper_id: str = "all"):
    # 👇 Accept without any condition (development mode)
    await websocket.accept()
    await handle_websocket(websocket, paper_id)

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Paper RAG System...")
    print(f"📚 Supported languages: {', '.join(config.SUPPORTED_LANGUAGES)}")
    print(f"🤖 LLM: Groq ({config.GROQ_MODEL})")
    print(f"💾 Cache: Upstash Redis")
    # Check vector DB
    try:
        info = vector_store.get_collection_info()
        print(f"📦 Vector DB: {info['points_count']} chunks indexed")
    except Exception as e:
        print(f"⚠️ Vector DB not ready: {e}")
    print("✅ Ready!")

@app.get("/")
async def root():
    return {
        "name": "Parth Thaker — Paper RAG System",
        "version": "2.0.0",
        "languages": config.SUPPORTED_LANGUAGES,
        "llm": config.GROQ_MODEL,
        "endpoints": {
            "health": "/api/health",
            "query": "/api/query",
            "websocket": "/ws/{paper_id}"
        }
    }