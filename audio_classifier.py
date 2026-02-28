import os
import tempfile
from groq import Groq
from scorer import detect_spam_in_text

# Get free API key at: https://console.groq.com
# Free tier: 7200 seconds of audio per day — plenty for MVP
client = Groq(api_key=os.getenv("GROQ_API_KEY", "your-groq-key-here"))

def transcribe_audio(audio_bytes: bytes, filename: str = "call_audio.wav") -> dict:
    """
    Transcribes audio using Groq's free Whisper API.
    Much faster than local Whisper and zero install size.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3",
                language="hi",        # Hindi first, falls back to auto
                response_format="text"
            )

        transcript = str(transcription).strip()
        spam_result = detect_spam_in_text(transcript)

        return {
            "transcript": transcript,
            "language": "auto-detected",
            "is_spam": spam_result["is_spam"],
            "confidence": spam_result["confidence"],
            "category": spam_result["category"],
            "keywords_found": spam_result["keywords_found"]
        }

    except Exception as e:
        return {
            "transcript": "",
            "language": "unknown",
            "is_spam": False,
            "confidence": 0.0,
            "category": "unknown",
            "keywords_found": [],
            "error": str(e)
        }

    finally:
        import os as _os
        try:
            _os.unlink(tmp_path)
        except:
            pass

def analyze_audio_file(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return transcribe_audio(f.read())