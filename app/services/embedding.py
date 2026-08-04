# ============================================
# EMBEDDING SERVICE — HF INFERENCE API (LOW MEMORY)
# ============================================
#
# WHY THIS CHANGE:
# The old version loaded `sentence-transformers` (which pulls in torch)
# at import time. That put the entire embedding model into RAM the
# moment the process started — before any request even arrived.
# On Render's free tier (512MB RAM), torch + the model alone can
# exceed the limit, so the process gets OOM-killed.
#
# This version calls Hugging Face's hosted Inference API over HTTP
# instead of loading any model locally. No torch, no local weights,
# no big RAM footprint. Trade-off: each embedding call now depends on
# network latency and HF's free-tier rate limits — the on-disk cache
# below exists specifically to blunt that cost for repeated queries.

import os
import requests
import numpy as np
import hashlib
import time

# ============================================
# CONFIG
# ============================================
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # set this in Render's Environment tab
HF_MODEL_ID = os.environ.get("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_ID}"

if not HF_API_TOKEN:
    raise RuntimeError(
        "HF_API_TOKEN environment variable is not set. "
        "Get a free token at https://huggingface.co/settings/tokens "
        "and add it in Render's Environment tab."
    )

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Retry settings — HF's free inference endpoints can return 503 while
# a model is "cold" (spinning up on their side). We retry with backoff
# instead of failing the request outright.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s

# ============================================
# CACHE SETUP
# ============================================
# Unchanged from before — this cache lives on disk, not RAM, so it
# was never the source of the memory problem. Keeping it because it
# also reduces the number of HF API calls (saves rate-limit budget).
CACHE_DIR = "data/cache/embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_key(texts):
    combined = "||".join(texts)
    return hashlib.md5(combined.encode()).hexdigest()


def _get_cached_embeddings(key):
    cache_file = os.path.join(CACHE_DIR, f"{key}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file).tolist()
    return None


def _save_embeddings(key, embeddings):
    cache_file = os.path.join(CACHE_DIR, f"{key}.npy")
    np.save(cache_file, np.array(embeddings))


# ============================================
# HF INFERENCE API CALL
# ============================================
def _call_hf_api(texts):
    """
    Sends texts to HF's feature-extraction endpoint and returns raw
    embeddings. Handles the two response shapes HF can return:
      - one embedding vector per text  -> [[...], [...], ...]
      - one token-level matrix per text -> [[[...], [...]], ...]
        (rare for e5-small, but some models return per-token vectors;
        we mean-pool those down to a single vector per text)
    """
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True},  # let HF queue the request instead of erroring while cold
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.exceptions.RequestException as e:
            last_error = str(e)

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            print(f"⚠️ HF API call failed ({last_error}), retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"HF Inference API failed after {MAX_RETRIES} attempts: {last_error}")


def _normalize_response(raw):
    """
    Converts whatever shape HF returned into a flat list of 1D vectors,
    one per input text. Mean-pools token-level output if needed.
    """
    result = []
    for item in raw:
        arr = np.array(item)
        if arr.ndim == 1:
            # already a single embedding vector
            result.append(arr)
        elif arr.ndim == 2:
            # token-level output: [num_tokens, hidden_dim] -> mean-pool to [hidden_dim]
            result.append(arr.mean(axis=0))
        else:
            raise ValueError(f"Unexpected embedding shape from HF API: {arr.shape}")
    return result


def _normalize_embeddings(vectors):
    """L2-normalize each vector so cosine similarity == dot product downstream."""
    normalized = []
    for v in vectors:
        norm = np.linalg.norm(v)
        normalized.append((v / norm).tolist() if norm > 0 else v.tolist())
    return normalized


# ============================================
# MAIN EMBEDDING FUNCTION
# ============================================
def get_embeddings(texts):
    """
    Generate embeddings for a list of texts via the HF Inference API,
    with on-disk caching so repeated queries don't re-hit the network.

    Returns: list of embedding vectors (list of floats), one per input text.
    Returns [] if texts is empty or if the API call ultimately fails.
    """
    if not texts:
        return []

    cache_key = _get_cache_key(texts)
    cached = _get_cached_embeddings(cache_key)
    if cached is not None:
        print(f"✅ Using cached embeddings for {len(texts)} texts")
        return cached

    print(f"🔄 Generating embeddings for {len(texts)} texts via HF Inference API...")

    try:
        raw = _call_hf_api(texts)
        vectors = _normalize_response(raw)
        embeddings_list = _normalize_embeddings(vectors)
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return []

    _save_embeddings(cache_key, embeddings_list)
    print(f"✅ Generated {len(embeddings_list)} embeddings")
    return embeddings_list