import google.generativeai as genai
import os

# Get free API key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAzIB0rodAieTLsNKByABb1yTsGrWz9qNA")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")  # Free tier model

GATEKEEPER_SYSTEM_PROMPT = """
You are an AI call screening assistant for an Indian mobile user.
Your job is to answer calls on behalf of the user and determine if the caller is legitimate or a spam/promotional caller.

Rules:
1. Greet the caller politely in English or Hindi
2. Ask them to state their name and reason for calling
3. If they mention any promotional content (loans, insurance, credit cards, offers, schemes), politely tell them the user is registered on DND and end the call
4. If they seem to be a genuine personal caller (friend, family, service provider), tell them you'll connect them now
5. Keep responses SHORT — under 2 sentences
6. Be firm but polite

Respond with a JSON like this:
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
    # Build conversation context
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['text']}"
        for msg in conversation_history
    ])

    prompt = f"""
{GATEKEEPER_SYSTEM_PROMPT}

Conversation so far:
{history_text}

Caller just said: "{caller_statement}"

What should the assistant say and do next?
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON from response
        import json
        import re
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
        "Namaste! This call is being screened by an automated assistant. "
        "Please state your name and the reason for your call."
    )