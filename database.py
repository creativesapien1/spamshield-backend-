from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# SQLite - free, local, no setup needed
DATABASE_URL = "sqlite:///./spamshield.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Table 1: Every phone number we've seen
class PhoneNumber(Base):
    __tablename__ = "phone_numbers"
    id = Column(Integer, primary_key=True, index=True)
    number = Column(String, unique=True, index=True)
    spam_score = Column(Float, default=0.0)       # 0.0 = clean, 1.0 = definite spam
    report_count = Column(Integer, default=0)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_confirmed_spam = Column(Boolean, default=False)

# Table 2: Individual spam reports from users
class SpamReport(Base):
    __tablename__ = "spam_reports"
    id = Column(Integer, primary_key=True, index=True)
    number = Column(String, index=True)
    reported_by = Column(String)                  # anonymous device ID
    category = Column(String)                     # bank / insurance / loan / other
    call_duration_seconds = Column(Integer)
    time_of_call = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

# Table 3: Audio analysis results
class AudioAnalysis(Base):
    __tablename__ = "audio_analysis"
    id = Column(Integer, primary_key=True, index=True)
    number = Column(String, index=True)
    transcript = Column(String)
    detected_language = Column(String)
    spam_keywords_found = Column(String)          # comma-separated
    confidence = Column(Float)
    analyzed_at = Column(DateTime, default=datetime.utcnow)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()