# app/config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = BASE_DIR / "models"

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
