from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
import uvicorn

from database import get_db, create_tables, PhoneNumber, SpamReport, AudioAnalysis
from scorer import calculate_behaviour_score, detect_spam_in_text
from audio_classifier import transcribe_audio
from gatekeeper import screen_caller, get_opening_greeting

app = FastAPI(
    title="SpamShield API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Allow React dashboard and Android app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "✅ SpamShield API is live",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Create database tables on startup
@app.on_event("startup")
def startup():
    create_tables()
    print("✅ SpamShield backend running!")

# ─────────────────────────────────────────────
# ENDPOINT 1: Check a number before call connects
# ─────────────────────────────────────────────
class CheckNumberRequest(BaseModel):
    number: str
    is_in_contacts: bool = False
    call_time: Optional[str] = None  # ISO format datetime string

@app.post("/api/check-number")
def check_number(req: CheckNumberRequest, db: Session = Depends(get_db)):
    # Get community report count from DB
    db_number = db.query(PhoneNumber).filter(PhoneNumber.number == req.number).first()
    report_count = db_number.report_count if db_number else 0

    call_time = datetime.fromisoformat(req.call_time) if req.call_time else datetime.utcnow()

    result = calculate_behaviour_score(
        number=req.number,
        call_time=call_time,
        is_in_contacts=req.is_in_contacts,
        report_count=report_count
    )

    # Save/update number in DB
    if not db_number:
        db_number = PhoneNumber(number=req.number, spam_score=result["spam_score"])
        db.add(db_number)
    else:
        db_number.spam_score = result["spam_score"]
        db_number.last_seen = datetime.utcnow()
    db.commit()

    return result

# ─────────────────────────────────────────────
# ENDPOINT 2: Submit a spam report (community feature)
# ─────────────────────────────────────────────
class SpamReportRequest(BaseModel):
    number: str
    reported_by: str         # anonymous device ID
    category: str            # bank/loan/insurance/other
    call_duration_seconds: int
    call_time: Optional[str] = None

@app.post("/api/report-spam")
def report_spam(req: SpamReportRequest, db: Session = Depends(get_db)):
    # Save the report
    report = SpamReport(
        number=req.number,
        reported_by=req.reported_by,
        category=req.category,
        call_duration_seconds=req.call_duration_seconds,
        time_of_call=datetime.fromisoformat(req.call_time) if req.call_time else datetime.utcnow()
    )
    db.add(report)

    # Update the number's score and report count
    db_number = db.query(PhoneNumber).filter(PhoneNumber.number == req.number).first()
    if not db_number:
        db_number = PhoneNumber(number=req.number, report_count=1)
        db.add(db_number)
    else:
        db_number.report_count += 1
        if db_number.report_count >= 5:
            db_number.is_confirmed_spam = True

    db.commit()
    return {"status": "reported", "message": "Thank you! This helps protect the community."}

# ─────────────────────────────────────────────
# ENDPOINT 3: Analyze call audio (10-sec clip)
# ─────────────────────────────────────────────
@app.post("/api/analyze-audio")
async def analyze_audio(
    number: str,
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    audio_bytes = await audio.read()

    # Run Whisper transcription + spam detection
    result = transcribe_audio(audio_bytes)

    # Save analysis to DB
    analysis = AudioAnalysis(
        number=number,
        transcript=result["transcript"],
        detected_language=result["language"],
        spam_keywords_found=",".join(result["keywords_found"]),
        confidence=result["confidence"]
    )
    db.add(analysis)

    # If spam detected, auto-increment report count
    if result["is_spam"]:
        db_number = db.query(PhoneNumber).filter(PhoneNumber.number == number).first()
        if not db_number:
            db_number = PhoneNumber(number=number, report_count=1, spam_score=result["confidence"])
            db.add(db_number)
        else:
            db_number.report_count += 1
            db_number.spam_score = max(db_number.spam_score, result["confidence"])
    db.commit()

    return result

# ─────────────────────────────────────────────
# ENDPOINT 4: AI Gatekeeper conversation
# ─────────────────────────────────────────────
class GatekeeperRequest(BaseModel):
    caller_statement: str
    conversation_history: list = []

@app.get("/api/gatekeeper/greeting")
def gatekeeper_greeting():
    return {"greeting": get_opening_greeting()}

@app.post("/api/gatekeeper/respond")
def gatekeeper_respond(req: GatekeeperRequest):
    result = screen_caller(req.conversation_history, req.caller_statement)
    return result

# ─────────────────────────────────────────────
# ENDPOINT 5: Dashboard stats
# ─────────────────────────────────────────────
@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total_numbers = db.query(PhoneNumber).count()
    confirmed_spam = db.query(PhoneNumber).filter(PhoneNumber.is_confirmed_spam == True).count()
    total_reports = db.query(SpamReport).count()
    total_analyses = db.query(AudioAnalysis).count()

    # Category breakdown
    from sqlalchemy import func
    categories = db.query(
        SpamReport.category,
        func.count(SpamReport.id).label("count")
    ).group_by(SpamReport.category).all()

    return {
        "total_numbers_tracked": total_numbers,
        "confirmed_spam_numbers": confirmed_spam,
        "total_community_reports": total_reports,
        "total_audio_analyses": total_analyses,
        "spam_categories": {cat: count for cat, count in categories}
    }

@app.get("/api/recent-reports")
def recent_reports(limit: int = 20, db: Session = Depends(get_db)):
    reports = db.query(SpamReport).order_by(SpamReport.created_at.desc()).limit(limit).all()
    return [
        {
            "number": r.number,
            "category": r.category,
            "duration": r.call_duration_seconds,
            "time": r.created_at.isoformat()
        }
        for r in reports
    ]

@app.get("/")
def root():
    return {
        "status": "SpamShield API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/api/check-number",
            "/api/report-spam", 
            "/api/analyze-audio",
            "/api/gatekeeper/greeting",
            "/api/gatekeeper/respond",
            "/api/stats",
            "/api/recent-reports"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
