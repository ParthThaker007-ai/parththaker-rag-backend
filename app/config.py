import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "papers"
    
    # Upstash Redis
    REDIS_URL = os.getenv("REDIS_URL")
    
    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    
    # Generation
    MAX_TOKENS = 512
    TEMPERATURE = 0.3
    
    # Languages
    SUPPORTED_LANGUAGES = ["en", "de", "fr"]
    LANGUAGE_NAMES = {
        "en": "English",
        "de": "German", 
        "fr": "French"
    }
    LANGUAGE_FLAGS = {
        "en": "🇬🇧",
        "de": "🇩🇪",
        "fr": "🇫🇷"
    }
    
    # Translation models
    TRANSLATION_MODELS = {
        ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    }
    
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    CACHE_TTL = 86400  # 24 hours

config = Config()