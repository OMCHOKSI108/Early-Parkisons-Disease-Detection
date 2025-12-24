# app/config.py

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]

# Allow overriding models directory via env for flexibility in containers
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

# Optional: if MODEL_S3_BUCKET is provided, models will be downloaded from S3 at startup
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
MODEL_S3_PREFIX = os.getenv("MODEL_S3_PREFIX", "")

# Drawing (spiral) models
DRAWING_MODELS_DIR = MODELS_DIR / "spiral"
HOG_SVM_PATH = DRAWING_MODELS_DIR / "parkinson_hog_svm_model.joblib"
HOG_SCALER_PATH = DRAWING_MODELS_DIR / "parkinson_hog_scaler.joblib"
# (spiral_parkinson_model_final.keras is optional / experimental)

# Voice models
VOICE_MODELS_DIR = MODELS_DIR / "voice"
VOICE_CSV_MODEL_PATH = VOICE_MODELS_DIR / "model_voice_csv_primary.keras"
VOICE_CSV_SCALER_PATH = VOICE_MODELS_DIR / "model_voice_csv_scaler.joblib"
VOICE_CSV_COLUMNS_PATH = VOICE_MODELS_DIR / "model_voice_csv_columns.joblib"

# Optional audio CNN
VOICE_AUDIO_SPEC_MODEL_PATH = VOICE_MODELS_DIR / "model_voice_audio_spec_secondary.keras"

# MFCC baseline
VOICE_AUDIO_MFCC_MODEL_PATH = VOICE_MODELS_DIR / "model_voice_audio_mfcc_baseline.keras"
VOICE_AUDIO_MFCC_SCALER_PATH = VOICE_MODELS_DIR / "model_voice_audio_mfcc_scaler.joblib"

# List of all model artifacts expected by the app (used by startup model downloader)
EXPECTED_MODEL_FILES = [
	# drawing
	str(HOG_SVM_PATH.name),
	str(HOG_SCALER_PATH.name),
	# voice csv
	str(VOICE_CSV_MODEL_PATH.name),
	str(VOICE_CSV_SCALER_PATH.name),
	str(VOICE_CSV_COLUMNS_PATH.name),
	# voice audio
	str(VOICE_AUDIO_SPEC_MODEL_PATH.name),
	str(VOICE_AUDIO_MFCC_MODEL_PATH.name),
	str(VOICE_AUDIO_MFCC_SCALER_PATH.name)
]
