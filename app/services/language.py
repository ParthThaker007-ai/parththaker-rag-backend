# ============================================
# LANGUAGE SERVICE — Using langdetect (No fasttext)
# ============================================

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import config
import os
import requests
import logging

# ============================================
# LANGDETECT (Replaces fasttext)
# ============================================
try:
    from langdetect import detect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # for reproducible results
    print("✅ Language detection using langdetect")
except ImportError:
    print("⚠️ langdetect not installed. Falling back to simple detection.")
    def detect(text):
        return 'en'

# ============================================
# TRANSLATION MODELS (Opus-MT via HuggingFace)
# ============================================
_translation_models = {}
_translation_tokenizers = {}
_translation_lock = None

# Try to use threading lock if available
try:
    import threading
    _translation_lock = threading.Lock()
except ImportError:
    _translation_lock = None

# Cache directory for models
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODEL_CACHE_DIR, "transformers")

# ============================================
# DETECT LANGUAGE
# ============================================
def detect_language(text: str) -> str:
    """
    Detect language using langdetect.
    Returns: 'en', 'de', 'fr', or 'en' as fallback.
    """
    if not text or not text.strip():
        return 'en'
    
    try:
        lang = detect(text)
        # langdetect returns 'en', 'de', 'fr', etc.
        # We only support en, de, fr, so if it's something else, default to 'en'
        if lang in ['en', 'de', 'fr']:
            return lang
        else:
            # Could be 'es', 'it', etc. Default to English
            return 'en'
    except Exception as e:
        print(f"⚠️ Language detection failed: {e}. Defaulting to 'en'.")
        return 'en'

# ============================================
# LOAD TRANSLATION MODEL
# ============================================
def get_translation_model(src: str, tgt: str):
    """
    Thread-safe loading of translation model.
    Uses lock if available to prevent duplicate downloads.
    """
    key = (src, tgt)
    
    # Check if already loaded in memory
    if key in _translation_models:
        return _translation_tokenizers[key], _translation_models[key]
    
    # Thread-safe loading
    if _translation_lock:
        with _translation_lock:
            if key in _translation_models:
                return _translation_tokenizers[key], _translation_models[key]
            
            return _load_translation_model(key, src, tgt)
    else:
        return _load_translation_model(key, src, tgt)

def _load_translation_model(key, src, tgt):
    """Internal function to actually load the model."""
    model_name = config.TRANSLATION_MODELS.get(key)
    if not model_name:
        raise ValueError(f"No translation model for {src}→{tgt}")
    
    print(f"📥 Loading translation model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=os.path.join(MODEL_CACHE_DIR, "transformers")
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=os.path.join(MODEL_CACHE_DIR, "transformers")
        )
        _translation_tokenizers[key] = tokenizer
        _translation_models[key] = model
        print(f"✅ Translation model loaded: {model_name}")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        raise

# ============================================
# TRANSLATE
# ============================================
def translate(text: str, src: str, tgt: str) -> str:
    """
    Translate text from src language to tgt using Opus-MT.
    Returns original text if translation fails.
    """
    if src == tgt:
        return text
    if not text or not text.strip():
        return text
    
    try:
        tokenizer, model = get_translation_model(src, tgt)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
    except Exception as e:
        print(f"❌ Translation error ({src}→{tgt}): {e}")
        return text

# ============================================
# CHECK SUPPORTED LANGUAGE
# ============================================
def is_supported_language(lang: str) -> bool:
    return lang in config.SUPPORTED_LANGUAGES