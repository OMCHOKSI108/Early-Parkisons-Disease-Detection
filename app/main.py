# app/main.py

import os
import warnings
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure the repository root is on sys.path so `import app.*` works
# when running `python main.py` from inside the `app/` folder.
import sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi import Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.drawing_model import drawing_model
from app.voice_model import voice_model
from app.database import (
    get_db, init_db, User, History, 
    hash_password, verify_password
)
from app.model_loader import ensure_models_present

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the application")
    # Ensure models are available (download from S3 if configured)
    try:
        ensure_models_present()
    except Exception as e:
        logger.warning("Model loader encountered an issue: %s", e)

    init_db()
    yield
    # Cleanup if needed

app = FastAPI(
    title="Parkinson's Disease Screening API",
    version="1.0.0",
    description="Drawing + Voice based Parkinson's screening models",
    lifespan=lifespan
)


if __name__ == "__main__":
    # Allow running via `python main.py` inside the `app/` folder
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")

# NEW: serve CSS / JS using absolute paths so running from `app/` works
app_dir = Path(__file__).resolve().parent
static_dir = app_dir / "static"
templates_dir = app_dir / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning("Static directory not found at %s, skipping mount", static_dir)

# Jinja2 templates
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    logger.warning("Templates directory not found at %s", templates_dir)
    templates = Jinja2Templates(directory=str(templates_dir))

# Usage limits configuration
USAGE_LIMITS = {"drawing": 10, "voice": 10}
LIMIT_RESET_HOURS = 3


@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


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
    
    logger.info(f"New user signed up: {username} ({email})")
    
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
    
    logger.info(f"User logged in: {user.username}")
    
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
async def get_usage(user_id: Optional[int] = Header(None), db: Session = Depends(get_db)):
    if user_id is None:
        user = get_or_create_guest(db)
    else:
        user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    time_until_reset = timedelta(hours=LIMIT_RESET_HOURS) - (datetime.utcnow() - user.last_reset)
    
    # Get user history
    history_records = db.query(History).filter(
        History.user_id == user.id
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
        "history": history_list,
        "user": {
            "id": user.id,
            "name": user.name,
            "username": user.username
        }
    }


def check_user_limit(user_id: Optional[int], prediction_type: str, db: Session) -> User:
    # Allow anonymous usage: if no user_id provided, create or reuse a guest user
    if user_id is None:
        user = get_or_create_guest(db)
    else:
        user = db.query(User).filter(User.id == user_id).first()

    if not user:
        # Fallback to guest user if provided id is invalid
        user = get_or_create_guest(db)
    
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
    return user


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


def get_or_create_guest(db: Session) -> User:
    """Return an existing guest user or create one for anonymous access."""
    guest = db.query(User).filter(User.username == "guest").first()
    if guest:
        return guest

    # Create minimal guest account
    guest = User(
        name="Guest",
        username="guest",
        email="guest@local",
        password_hash=hash_password("guest")
    )
    db.add(guest)
    db.commit()
    db.refresh(guest)
    return guest


# ---------- Drawing endpoint ----------



# ... keep the existing endpoints ...

@app.post("/predict/voice/audio")
async def predict_voice_audio(
    file: UploadFile = File(...), 
    user_id: Optional[int] = Header(None), 
    db: Session = Depends(get_db)
):
    """
    Predict Parkinson vs Healthy using ONLY the audio CNN model.
    Upload a .wav file.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV audio file")
    
    # Check usage limit (returns resolved user)
    user = check_user_limit(user_id, "voice", db)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = Path(tmp_file.name)

    try:
        result = voice_model.predict_from_audio_file(tmp_path)
        add_to_history(user.id, "voice", result, "voice_audio_cnn", db)
    except Exception as e:
        logger.error(f"Error in voice prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"model": "voice_audio_cnn", **result}


@app.post("/predict/drawing")
async def predict_drawing(
    file: UploadFile = File(...), 
    user_id: Optional[int] = Header(None), 
    db: Session = Depends(get_db)
):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file (PNG, JPG, JPEG)")
    
    # Check usage limit (returns resolved user)
    user = check_user_limit(user_id, "drawing", db)

    file_bytes = await file.read()
    try:
        result = drawing_model.predict_from_bytes(file_bytes)
        add_to_history(user.id, "drawing", result, "drawing_hog_svm", db)
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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = Path(tmp_file.name)

    try:
        result = voice_model.predict_ensemble(features, tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in voice ensemble prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"model": "voice_csv+audio_ensemble", **result}


@app.post("/predict/voice/final-audio")
async def predict_voice_final_audio(
    file: UploadFile = File(...), 
    user_id: Optional[int] = Header(None), 
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
    
    # Check usage limit (returns resolved user)
    user = check_user_limit(user_id, "voice", db)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = Path(tmp_file.name)

    try:
        result = voice_model.predict_final_audio(tmp_path)
        add_to_history(user.id, "voice", result, "voice_audio_final_ensemble", db)
    except Exception as e:
        logger.error(f"Error in final voice prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"model": "voice_audio_final_ensemble", **result}
