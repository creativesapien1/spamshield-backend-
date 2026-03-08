import google.generativeai as genai
import os
import json
import re

# API key must be set as environment variable on Render
# Get a free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)

# gemini-2.0-flash — free tier, 15 req/min, 1500/day
model = genai.GenerativeModel("gemini-2.0-flash")

GATEKEEPER_SYSTEM_PROMPT = """
You are an AI call screening assistant for an Indian mobile user.
Your job is to answer calls on behalf of the user and determine if the caller is legitimate or spam.

Rules:
1. Greet the caller politely in English
2. Ask them to state their name and reason for calling
3. If they mention any promotional content (loans, insurance, credit cards, offers, schemes, investments), action must be BLOCK
4. If they seem to be a genuine personal caller (friend, family, service provider with appointment), action must be CONNECT
5. Keep response_text SHORT — under 2 sentences
6. Be firm but polite

You MUST respond with valid JSON only, no markdown, no extra text:
{
  "response_text": "what to say to the caller",
  "action": "CONNECT" or "BLOCK" or "CONTINUE_SCREENING",
  "reason": "brief explanation"
}
"""

def screen_caller(conversation_history: list, caller_statement: str) -> dict:
    """
    Takes what the caller said and decides what to do.
    conversation_history: list of {"role": "caller"/"assistant", "text": "..."}
    """
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['text']}"
        for msg in conversation_history
    ])

    prompt = f"""{GATEKEEPER_SYSTEM_PROMPT}

Conversation so far:
{history_text}

Caller just said: "{caller_statement}"

Respond with JSON only:"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Strip markdown code fences if present
        text = re.sub(r'```json|```', '', text).strip()

        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {
                "response_text": "Please hold, connecting you now.",
                "action": "CONNECT",
                "reason": "Could not parse AI response"
            }

    except Exception as e:
        return {
            "response_text": "Please hold, connecting you now.",
            "action": "CONNECT",
            "reason": f"Error: {str(e)}"
        }

def get_opening_greeting() -> str:
    """What the AI says when it first picks up."""
    return (
        "Namaste! This call is being screened by SpamShield. "
        "Please state your name and the reason for your call."
    )