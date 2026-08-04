import redis
import json
import hashlib
from app.config import config

_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is None:
        if config.REDIS_URL:
            try:
                _redis_client = redis.from_url(
                    config.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                _redis_client.ping()
                print("✅ Upstash Redis connected.")
            except Exception as e:
                print(f"⚠️ Upstash Redis connection failed: {e}")
                _redis_client = None
        else:
            print("⚠️ No Redis URL provided. Cache disabled.")
    return _redis_client

def get_cache_key(query: str, paper_id: str = None, language: str = "en") -> str:
    key_data = f"{query}|{paper_id or 'all'}|{language}"
    return f"rag:query:{hashlib.md5(key_data.encode()).hexdigest()}"

def get_cached_response(query: str, paper_id: str = None, language: str = "en"):
    client = get_redis_client()
    if not client:
        return None
    
    key = get_cache_key(query, paper_id, language)
    try:
        cached = client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"⚠️ Cache read error: {e}")
    return None

def cache_response(query: str, response: dict, paper_id: str = None, language: str = "en"):
    client = get_redis_client()
    if not client:
        return
    
    key = get_cache_key(query, paper_id, language)
    try:
        client.setex(key, config.CACHE_TTL, json.dumps(response, default=str))
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

def clear_cache():
    client = get_redis_client()
    if not client:
        return
    
    try:
        keys = client.keys("rag:query:*")
        if keys:
            client.delete(*keys)
            print(f"✅ Cleared {len(keys)} cache entries.")
    except Exception as e:
        print(f"⚠️ Cache clear failed: {e}")