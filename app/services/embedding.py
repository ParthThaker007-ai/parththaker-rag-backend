# ============================================
# EMBEDDING SERVICE — LOCAL & ROBUST
# ============================================

import os
import sys
import numpy as np
import hashlib
import time
import json

# ============================================
# FIX: Python 3.11+ security bypass
# ============================================
os.environ["PYTHONSAFEPATH"] = "1"

# ============================================
# TRY SENTENCE-TRANSFORMERS
# ============================================
USE_SENTENCE_TRANSFORMERS = False
embedding_model = None

try:
    from sentence_transformers import SentenceTransformer
    print("📥 Loading local embedding model: intfloat/multilingual-e5-small")
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
    USE_SENTENCE_TRANSFORMERS = True
    print("✅ Local embedding model loaded successfully.")
except Exception as e:
    print(f"⚠️ SentenceTransformers failed: {e}")
    print("📥 Trying transformers fallback...")
    
    # ============================================
    # FALLBACK: TRANSFORMERS PIPELINE
    # ============================================
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import torch.nn.functional as F
        
        model_name = "intfloat/multilingual-e5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        USE_SENTENCE_TRANSFORMERS = False
        print("✅ Transformers model loaded.")
    except Exception as e2:
        print(f"❌ Transformers fallback failed: {e2}")
        print("📥 Trying MiniLM as last resort...")
        try:
            from transformers import AutoTokenizer, AutoModel
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            USE_SENTENCE_TRANSFORMERS = False
            print("✅ MiniLM loaded as last resort.")
        except Exception as e3:
            raise RuntimeError(f"No embedding model available: {e3}")

# ============================================
# CACHE SETUP
# ============================================
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
# MAIN EMBEDDING FUNCTION
# ============================================
def get_embeddings(texts):
    """Generate embeddings using the available model."""
    if not texts:
        return []
    
    cache_key = _get_cache_key(texts)
    cached = _get_cached_embeddings(cache_key)
    if cached is not None:
        print(f"✅ Using cached embeddings for {len(texts)} texts")
        return cached
    
    print(f"🔄 Generating embeddings for {len(texts)} texts locally...")
    
    try:
        if USE_SENTENCE_TRANSFORMERS:
            # Sentence-Transformers
            embeddings = embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
        else:
            # Transformers pipeline
            embeddings_list = []
            with torch.no_grad():
                for text in texts:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    outputs = model(**inputs)
                    # Mean pooling
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    embeddings_list.append(emb.tolist())
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return []
    
    _save_embeddings(cache_key, embeddings_list)
    print(f"✅ Generated {len(embeddings_list)} embeddings")
    return embeddings_list