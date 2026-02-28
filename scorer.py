from datetime import datetime
from typing import Optional

# Spam keywords across Hindi + English (expandable)
SPAM_PHRASES = [
    # English
    "loan", "insurance", "credit card", "pre-approved", "offer", "scheme",
    "investment", "policy", "emi", "interest rate", "bank", "congratulations",
    "selected", "winner", "prize", "kyc", "account", "demat", "mutual fund",
    # Hindi (romanized - common in Indian call centers)
    "loan milega", "bima", "jeevan", "yojana", "nivesh", "fayde",
    # Common openers
    "i'm calling from", "main bol raha hoon", "aapko ek offer",
    "special offer", "limited time", "free", "zero cost", "no emi"
]

SPAM_CATEGORIES = {
    "bank": ["bank", "credit card", "account", "kyc", "demat", "savings"],
    "loan": ["loan", "emi", "interest", "borrow", "credit", "finance"],
    "insurance": ["insurance", "bima", "policy", "premium", "life cover", "term plan"],
    "investment": ["mutual fund", "sip", "stock", "investment", "demat", "trading"],
    "scam": ["winner", "prize", "congratulations", "selected", "lottery", "free gift"]
}

def calculate_behaviour_score(
    number: str,
    call_time: Optional[datetime] = None,
    is_in_contacts: bool = False,
    previous_call_count: int = 0,
    report_count: int = 0,
    call_duration_history: list = []
) -> dict:
    """
    Scores a phone number from 0.0 (clean) to 1.0 (spam)
    based purely on behavioural signals — no AI needed.
    """
    score = 0.0
    reasons = []

    # Signal 1: Not in contacts (weight: 0.2)
    if not is_in_contacts:
        score += 0.2
        reasons.append("Number not in contacts")

    # Signal 2: Calling during peak spam hours 10am-7pm weekdays (weight: 0.15)
    if call_time:
        hour = call_time.hour
        weekday = call_time.weekday()  # 0=Monday, 6=Sunday
        if 10 <= hour <= 19 and weekday <= 4:
            score += 0.15
            reasons.append("Called during peak spam hours (10am-7pm weekday)")

    # Signal 3: Community has reported this number (weight: up to 0.4)
    if report_count > 0:
        report_score = min(report_count * 0.1, 0.4)
        score += report_score
        reasons.append(f"Reported by {report_count} user(s) in community")

    # Signal 4: Short call history suggests robocall pattern (weight: 0.15)
    if call_duration_history:
        avg_duration = sum(call_duration_history) / len(call_duration_history)
        if avg_duration < 30:  # calls shorter than 30 seconds
            score += 0.15
            reasons.append("Previous calls from this number were very short (robocall pattern)")

    # Signal 5: New number never called before (weight: 0.1)
    if previous_call_count == 0:
        score += 0.1
        reasons.append("First time this number has called")

    # Cap at 1.0
    final_score = min(round(score, 2), 1.0)

    return {
        "number": number,
        "spam_score": final_score,
        "risk_level": get_risk_level(final_score),
        "reasons": reasons,
        "recommendation": get_recommendation(final_score)
    }

def get_risk_level(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def get_recommendation(score: float) -> str:
    if score >= 0.7:
        return "BLOCK — High probability spam call"
    elif score >= 0.4:
        return "SCREEN — Let AI gatekeeper answer first"
    else:
        return "ALLOW — Likely a genuine call"

def detect_spam_in_text(transcript: str) -> dict:
    """
    Checks a call transcript for spam keywords.
    Used after Whisper transcribes the audio.
    """
    transcript_lower = transcript.lower()
    found_keywords = []
    detected_category = "unknown"
    highest_match = 0

    for category, keywords in SPAM_CATEGORIES.items():
        matches = [kw for kw in keywords if kw in transcript_lower]
        if len(matches) > highest_match:
            highest_match = len(matches)
            detected_category = category
            found_keywords = matches

    # Also check general spam phrases
    general_matches = [phrase for phrase in SPAM_PHRASES if phrase in transcript_lower]
    all_keywords = list(set(found_keywords + general_matches))

    confidence = min(len(all_keywords) * 0.2, 1.0)

    return {
        "is_spam": confidence >= 0.4,
        "confidence": confidence,
        "category": detected_category,
        "keywords_found": all_keywords
    }