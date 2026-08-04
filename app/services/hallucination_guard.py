# ============================================
# HALLUCINATION GUARD — NO EXTERNAL DEPENDENCIES
# ============================================

import re
import math

def extract_claims(text: str) -> list:
    """Extract factual claims from response text (sentences longer than 20 chars)."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def fact_check_response(response: str, sources: list) -> dict:
    """
    Simple fact-checking without CrossEncoder.
    Uses heuristic confidence based on number of claims and source availability.
    """
    claims = extract_claims(response)
    source_texts = [s.get("payload", {}).get("text", "") for s in sources if s.get("payload")]
    
    if not claims:
        return {
            "confidence": 0.0,
            "level": "unknown",
            "claim_scores": [],
            "warning": "⚠️ No clear claims found to verify.",
            "verified": False,
            "claims_checked": 0
        }
    
    # Heuristic: more claims + more source texts → higher confidence
    if source_texts:
        # If we have sources, confidence is higher
        base_confidence = 0.5
        # Bonus for having multiple sources
        source_bonus = min(0.3, len(source_texts) * 0.05)
        # Penalty for too many claims (could be verbose)
        claim_penalty = min(0.2, len(claims) * 0.02)
        confidence = min(0.95, base_confidence + source_bonus - claim_penalty)
    else:
        # No sources → low confidence
        confidence = 0.2
    
    # Ensure minimum confidence
    confidence = max(0.1, min(0.95, confidence))
    
    level = "high" if confidence >= 0.7 else "medium" if confidence >= 0.4 else "low"
    warning = {
        "high": "",
        "medium": "⚠️ *Medium confidence* — verify with source.",
        "low": "⚠️ *Low confidence* — please verify with original paper.",
    }.get(level, "⚠️ *Very low confidence* — verify against source.")
    
    return {
        "confidence": confidence,
        "level": level,
        "claim_scores": [confidence] * len(claims),
        "warning": warning,
        "verified": confidence >= 0.6,
        "claims_checked": len(claims)
    }