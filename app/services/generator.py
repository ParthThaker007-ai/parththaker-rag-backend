import groq
from app.config import config

groq_client = groq.Client(api_key=config.GROQ_API_KEY)

# ============================================================
# MASTER SYSTEM PROMPT – Forces exact answers with citations
# ============================================================
SYSTEM_PROMPT = """
You are an expert research assistant. You answer questions based **ONLY** on the provided context from research papers.

### INSTRUCTIONS:
1. **If the question asks for a definition, formula, or specific detail** – quote the exact text from the context (including equations, numbers, and acronyms) and provide the source reference (e.g., [Source 1]).
2. **If the context does NOT contain the answer** – say clearly: "I cannot find this information in the provided excerpts." Do NOT make up anything.
3. **If the question is multi‑part** – break your answer into numbered points, each supported by a citation.
4. **If the context gives conflicting information** – mention both and explain the conflict.
5. **Always include citations** after each claim (e.g., [Source 3, page 5]).
6. **Keep your answer concise but complete.** Use bullet points for clarity when helpful.
7. **If the question is about a metric or model comparison** – extract the relevant numbers and comparisons from the context, and present them clearly.
8. **If you are unsure about a number** – check the context again; if not found, say "The exact value is not stated."

### FORMAT:
- Use clear, plain English.
- When quoting a formula, write it in LaTeX or plain text exactly as in the paper.
- End your answer with a summary of key takeaways (optional).
"""

def generate_response(query: str, context: list[str]) -> dict:
    """Generate a response using Groq with the master system prompt."""
    context_text = "\n\n".join([
        f"[Source {i+1}] {chunk[:800]}..." if len(chunk) > 800 else chunk
        for i, chunk in enumerate(context)
    ])

    user_prompt = f"""Context from paper excerpts:
{context_text}

Question: {query}

Answer (using ONLY the provided context, with citations):"""

    try:
        response = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,          # Lower temperature for factual responses
            max_tokens=600,
            top_p=0.9,
        )
        generated_text = response.choices[0].message.content
        return {
            "text": generated_text,
            "tokens": len(generated_text.split())
        }
    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        return {"text": f"Error: {e}", "tokens": 0}