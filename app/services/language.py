# ============================================
# LANGUAGE SERVICE — langdetect + Groq for translation
# ============================================
#
# WHY THIS VERSION EXISTS:
# The original version loaded Opus-MT (Helsinki-NLP) translation models
# via `transformers` + `torch`, imported at module level. Even though
# the actual model only loaded lazily on first use, the bare
# `import torch` at the top of the file meant the whole app crashed
# at startup once torch was removed from requirements.txt (to fix the
# earlier OOM issue). Re-adding torch just for this would reintroduce
# the same memory risk this project already fought twice.
#
# Instead, this version reuses Groq — the LLM already used for answer
# generation (see generator.py) — to do the translation via a plain
# prompt. Zero new dependencies, zero extra memory, and one less
# network domain to depend on. For short query/response text (which
# is all this app translates), LLM-based translation is a normal,
# production-viable approach — not a toy substitute for MT models.

import groq
from app.config import config

# ============================================
# LANGDETECT
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

groq_client = groq.Client(api_key=config.GROQ_API_KEY)

# Human-readable names for the prompt — an LLM translates better when
# told "German" than when told the raw ISO code "de".
_LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
}

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
# TRANSLATE (via Groq)
# ============================================
def translate(text: str, src: str, tgt: str) -> str:
    """
    Translate text from src language to tgt using Groq's LLM.
    Returns original text if translation fails, or if src == tgt.
    """
    if src == tgt:
        return text
    if not text or not text.strip():
        return text

    src_name = _LANGUAGE_NAMES.get(src, src)
    tgt_name = _LANGUAGE_NAMES.get(tgt, tgt)

    # Deliberately terse system prompt: we want ONLY the translation
    # back, no "Here is the translation:" preamble, no quotes, no
    # commentary — this text gets used directly downstream (embedded,
    # or shown to the user as the final answer).
    system_prompt = (
        f"You are a precise translator. Translate the user's {src_name} text "
        f"into {tgt_name}. Return ONLY the translated text — no explanations, "
        f"no quotation marks, no additional commentary."
    )

    try:
        response = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.1,  # low temperature — translation should be deterministic, not creative
            max_tokens=1024,
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print(f"❌ Translation error ({src}→{tgt}): {e}")
        return text

# ============================================
# CHECK SUPPORTED LANGUAGE
# ============================================
def is_supported_language(lang: str) -> bool:
    return lang in config.SUPPORTED_LANGUAGES