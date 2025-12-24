# app/database.py

import os
import logging
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# PostgreSQL connection from environment or default
DATABASE_URL = os.getenv("DATABASE_URL")

# Allow a developer-friendly SQLite fallback when DATABASE_URL is not provided.
# Set the environment variable `ALLOW_SQLITE_FALLBACK=false` to require DATABASE_URL.
allow_fallback = os.getenv("ALLOW_SQLITE_FALLBACK", "true").lower() in ("1", "true", "yes")

if not DATABASE_URL:
    if allow_fallback:
        dev_db_path = Path(__file__).resolve().parents[1] / "dev_database.db"
        DATABASE_URL = f"sqlite:///{dev_db_path}"
        logger.warning("DATABASE_URL not set; using local SQLite fallback at %s", dev_db_path)
    else:
        raise ValueError("DATABASE_URL environment variable is not set")

# For PostgreSQL production, support both postgres:// and postgresql:// schemes
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configure connect_args based on the backend
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
elif "sslmode" in DATABASE_URL:
    connect_args = {"sslmode": "require"}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    drawing_count = Column(Integer, default=0)
    voice_count = Column(Integer, default=0)
    last_reset = Column(DateTime, default=datetime.utcnow)


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    prediction_type = Column(String(50))  # 'drawing' or 'voice'
    predicted_label = Column(String(50))
    confidence = Column(Float)
    prob_pd_raw = Column(Float)
    model_name = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password"""
    return pwd_context.verify(plain_password, hashed_password)
