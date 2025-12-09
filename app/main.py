# app/main.py

import os
import warnings
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi import Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from sqlalchemy.orm import Session

from .drawing_model import drawing_model
from .voice_model import voice_model
from .database import (
    get_db, init_db, User, History, 
    hash_password, verify_password
)

app = FastAPI(
    title="Parkinson's Disease Screening API",
    version="1.0.0",
    description="Drawing + Voice based Parkinson's screening models",
)

# Initialize database
@app.on_event("startup")
def startup_event():
    init_db()

# NEW: serve CSS / JS
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# NEW: Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Usage limits configuration
USAGE_LIMITS = {"drawing": 10, "voice": 10}
LIMIT_RESET_HOURS = 3


@app.get("/")
def root():
    return {"message": "Parkinson's Screening API is running"}


@app.get("/ui", response_class=HTMLResponse)
def ui_page(request: Request):
    """
    Simple web UI to interact with drawing + voice models.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/auth/signup")
async def signup(credentials: Dict[str, str] = Body(...), db: Session = Depends(get_db)):
    name = credentials.get("name", "").strip()
    username = credentials.get("username", "").strip()
    email = credentials.get("email", "").strip()
    password = credentials.get("password", "").strip()
    
    if not all([name, username, email, password]):
        raise HTTPException(status_code=400, detail="All fields are required")
    
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == email) | (User.username == username)
    ).first()
    
    if existing_user:
        if existing_user.email == email:
            raise HTTPException(status_code=400, detail="Email already registered")
        else:
            raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new user
    new_user = User(
        name=name,
        username=username,
        email=email,
        password_hash=hash_password(password)
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "success": True,
        "message": "Account created successfully",
        "user": {
            "name": new_user.name,
            "username": new_user.username,
            "email": new_user.email
        }
    }


@app.post("/auth/login")
async def login(credentials: Dict[str, str] = Body(...), db: Session = Depends(get_db)):
    login_id = credentials.get("login_id", "").strip()  # email or username
    password = credentials.get("password", "").strip()
    
    if not login_id or not password:
        raise HTTPException(status_code=400, detail="Login credentials are required")
    
    # Find user by email or username
    user = db.query(User).filter(
        (User.email == login_id) | (User.username == login_id)
    ).first()
    
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if limits need reset
    if datetime.utcnow() - user.last_reset > timedelta(hours=LIMIT_RESET_HOURS):
        user.drawing_count = 0
        user.voice_count = 0
        user.last_reset = datetime.utcnow()
        db.commit()
    
    return {
        "success": True,
        "user": {
            "id": user.id,
            "name": user.name,
            "username": user.username,
            "email": user.email
        },
        "usage": {
            "drawing": user.drawing_count,
            "voice": user.voice_count,
            "limits": USAGE_LIMITS
        }
    }


@app.get("/auth/usage")
async def get_usage(user_id: int = Header(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    time_until_reset = timedelta(hours=LIMIT_RESET_HOURS) - (datetime.utcnow() - user.last_reset)
    
    # Get user history
    history_records = db.query(History).filter(
        History.user_id == user_id
    ).order_by(History.timestamp.desc()).limit(20).all()
    
    history_list = [{
        "type": h.prediction_type,
        "timestamp": h.timestamp.isoformat(),
        "result": {
            "predicted_label": h.predicted_label,
            "confidence": h.confidence,
            "prob_pd_raw": h.prob_pd_raw,
            "model": h.model_name
        }
    } for h in history_records]
    
    return {
        "drawing_count": user.drawing_count,
        "voice_count": user.voice_count,
        "limits": USAGE_LIMITS,
        "time_until_reset_minutes": max(0, int(time_until_reset.total_seconds() / 60)),
        "history": history_list
    }


def check_user_limit(user_id: int, prediction_type: str, db: Session):
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Please login first")
    
    # Check if limits need reset
    if datetime.utcnow() - user.last_reset > timedelta(hours=LIMIT_RESET_HOURS):
        user.drawing_count = 0
        user.voice_count = 0
        user.last_reset = datetime.utcnow()
        db.commit()
    
    count_key = f"{prediction_type}_count"
    current_count = getattr(user, count_key)
    
    if current_count >= USAGE_LIMITS[prediction_type]:
        time_until_reset = timedelta(hours=LIMIT_RESET_HOURS) - (datetime.utcnow() - user.last_reset)
        minutes = int(time_until_reset.total_seconds() / 60)
        raise HTTPException(
            status_code=429,
            detail=f"Usage limit reached. You can use this feature again in {minutes} minutes."
        )
    
    setattr(user, count_key, current_count + 1)
    db.commit()


def add_to_history(user_id: int, prediction_type: str, result: dict, model_name: str, db: Session):
    history_entry = History(
        user_id=user_id,
        prediction_type=prediction_type,
        predicted_label=result.get("predicted_label", result.get("final_label", "")),
        confidence=result.get("confidence", result.get("final_confidence", 0)),
        prob_pd_raw=result.get("prob_pd_raw", result.get("final_prob_pd", 0)),
        model_name=model_name
    )
    db.add(history_entry)
    db.commit()


# ---------- Drawing endpoint ----------



# ... keep the existing endpoints ...

@app.post("/predict/voice/audio")
async def predict_voice_audio(
    file: UploadFile = File(...), 
    user_id: int = Header(...), 
    db: Session = Depends(get_db)
):
    """
    Predict Parkinson vs Healthy using ONLY the audio CNN model.
    Upload a .wav file.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV audio file")
    
    # Check usage limit
    check_user_limit(user_id, "voice", db)

    tmp_path = Path("temp_voice.wav")
    tmp_path.write_bytes(await file.read())

    try:
        result = voice_model.predict_from_audio_file(tmp_path)
        add_to_history(user_id, "voice", result, "voice_audio_cnn", db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"model": "voice_audio_cnn", **result}


@app.post("/predict/drawing")
async def predict_drawing(
    file: UploadFile = File(...), 
    user_id: int = Header(...), 
    db: Session = Depends(get_db)
):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file (PNG, JPG, JPEG)")
    
    # Check usage limit
    check_user_limit(user_id, "drawing", db)

    file_bytes = await file.read()
    try:
        result = drawing_model.predict_from_bytes(file_bytes)
        add_to_history(user_id, "drawing", result, "drawing_hog_svm", db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"model": "drawing_hog_svm", **result}


# ---------- Voice endpoints ----------

@app.post("/predict/voice/csv")
async def predict_voice_csv(features: Dict[str, float] = Body(...)):
    """
    Body should be JSON mapping feature_name -> value,
    matching the columns in model_voice_csv_columns.joblib
    """
    try:
        result = voice_model.predict_from_csv_features(features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"model": "voice_csv_mlp", **result}


@app.post("/predict/voice/ensemble")
async def predict_voice_ensemble(
    features: Dict[str, float] = Body(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav audio file")

    tmp_path = Path("temp_audio.wav")
    tmp_path.write_bytes(await file.read())

    try:
        result = voice_model.predict_ensemble(features, tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"model": "voice_csv+audio_ensemble", **result}


@app.post("/predict/voice/final-audio")
async def predict_voice_final_audio(
    file: UploadFile = File(...), 
    user_id: int = Header(...), 
    db: Session = Depends(get_db)
):
    """
    Final audio-only endpoint.
    User uploads a .wav file, we use audio MFCC-MLP + audio CNN ensemble.
    NOTE: CSV model is NOT used here, because it needs pre-computed
    speech features that cannot be derived from wav with our current code.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV audio file")
    
    # Check usage limit
    check_user_limit(user_id, "voice", db)

    tmp_path = Path("temp_final_voice.wav")
    tmp_path.write_bytes(await file.read())

    try:
        result = voice_model.predict_final_audio(tmp_path)
        add_to_history(user_id, "voice", result, "voice_audio_final_ensemble", db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"model": "voice_audio_final_ensemble", **result}
