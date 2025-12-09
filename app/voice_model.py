# app/voice_model.py

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import librosa
import tensorflow as tf

from .config import (
    VOICE_CSV_MODEL_PATH,
    VOICE_CSV_SCALER_PATH,
    VOICE_CSV_COLUMNS_PATH,
    VOICE_AUDIO_SPEC_MODEL_PATH,
    VOICE_AUDIO_MFCC_MODEL_PATH,
    VOICE_AUDIO_MFCC_SCALER_PATH,
)


class VoiceModel:
    def __init__(self):
        # Load CSV-based model (primary)
        self.csv_model = tf.keras.models.load_model(VOICE_CSV_MODEL_PATH)
        self.csv_scaler = joblib.load(VOICE_CSV_SCALER_PATH)
        self.csv_columns: List[str] = joblib.load(VOICE_CSV_COLUMNS_PATH)

        # Optional audio CNN (secondary)
        try:
            self.audio_cnn = tf.keras.models.load_model(VOICE_AUDIO_SPEC_MODEL_PATH)
        except Exception:
            self.audio_cnn = None

        # Audio MFCC baseline – optional
        try:
            self.audio_mfcc_model = tf.keras.models.load_model(VOICE_AUDIO_MFCC_MODEL_PATH)
            self.audio_mfcc_scaler = joblib.load(VOICE_AUDIO_MFCC_SCALER_PATH)
        except Exception:
            self.audio_mfcc_model = None
            self.audio_mfcc_scaler = None

    # ---------- CSV part ----------

    def _vector_from_json(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert a dict {feature_name: value} into a numpy vector
        in the correct column order.
        """
        vals = []
        for col in self.csv_columns:
            if col not in features:
                raise ValueError(f"Missing feature '{col}' in request payload")
            vals.append(float(features[col]))
        return np.array(vals, dtype=np.float32)

    def predict_from_csv_features(self, features: Dict[str, float]) -> Dict:
        vec = self._vector_from_json(features).reshape(1, -1)
        vec_scaled = self.csv_scaler.transform(vec)
        prob_pd = self.csv_model.predict(vec_scaled)[0, 0]

        if prob_pd >= 0.5:
            label = "Parkinson"
            confidence = prob_pd
        else:
            label = "Healthy"
            confidence = 1.0 - prob_pd

        return {
            "predicted_label": label,
            "prob_pd_raw": float(prob_pd),
            "confidence": float(confidence),
        }

    # ---------- Audio CNN (optional) ----------

    @staticmethod
    def _mel_spectrogram_image(
        path: str,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        max_duration: float = 3.0,
        target_width: int = 128,
    ) -> np.ndarray:
        import librosa
        import numpy as np

        audio, _ = librosa.load(path, sr=sr)
        max_len = int(max_duration * sr)
        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            audio = np.pad(audio, (0, max_len - len(audio)))

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = librosa.util.fix_length(mel_db, size=target_width, axis=1)

        min_v, max_v = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - min_v) / (max_v - min_v + 1e-8)
        return mel_norm.astype("float32")

    @staticmethod
    def _mfcc_from_file(path: str, sr: int = 16000, n_mfcc: int = 40, max_duration: float = 3.0):
        try:
            audio, _ = librosa.load(path, sr=sr)
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {e}")

        max_len = int(max_duration * sr)
        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            audio = np.pad(audio, (0, max_len - len(audio)))

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc.mean(axis=1).astype("float32")

    def predict_from_audio_file(self, wav_path: Path) -> Dict:
        if self.audio_cnn is None:
            raise RuntimeError("Audio CNN model not loaded")

        spec = self._mel_spectrogram_image(str(wav_path))
        img = spec[..., np.newaxis]
        batch = np.expand_dims(img, 0)
        prob_pd = 1 - self.audio_cnn.predict(batch)[0, 0]  # Invert: model outputs prob_healthy

        if prob_pd >= 0.5:
            label = "Parkinson"
            confidence = prob_pd
        else:
            label = "Healthy"
            confidence = 1.0 - prob_pd

        return {
            "predicted_label": label,
            "prob_pd_raw": float(prob_pd),
            "confidence": float(confidence),
        }

    # ---------- Optional ensemble ----------

    def predict_ensemble(
        self,
        features: Dict[str, float],
        wav_path: Path,
        weights=(0.7, 0.3),
    ) -> Dict:
        """
        CSV primary + audio CNN secondary.
        """
        res_csv = self.predict_from_csv_features(features)
        p_csv = res_csv["prob_pd_raw"]

        if self.audio_cnn is None:
            # fallback: CSV only
            return {
                "csv_model": res_csv,
                "audio_model": None,
                "final_label": res_csv["predicted_label"],
                "final_prob_pd": res_csv["prob_pd_raw"],
                "final_confidence": res_csv["confidence"],
            }

        res_audio = self.predict_from_audio_file(wav_path)
        p_audio = res_audio["prob_pd_raw"]

        w_csv, w_audio = weights
        p_final = w_csv * p_csv + w_audio * p_audio

        if p_final >= 0.5:
            label = "Parkinson"
            conf = p_final
        else:
            label = "Healthy"
            conf = 1.0 - p_final

        return {
            "csv_model": res_csv,
            "audio_model": res_audio,
            "final_label": label,
            "final_prob_pd": float(p_final),
            "final_confidence": float(conf),
        }

    def predict_final_audio(self, wav_path: Path, weights=(0.5, 0.5)) -> Dict:
        """
        Final audio-only prediction.
        Uses MFCC-MLP and Spectrogram-CNN (if both available).
        weights: (w_mfcc, w_spec)
        """

        if self.audio_mfcc_model is None and self.audio_cnn is None:
            raise RuntimeError("No audio models loaded")

        probs = []
        parts = {}

        # MFCC MLP
        if self.audio_mfcc_model is not None and self.audio_mfcc_scaler is not None:
            mfcc_vec = self._mfcc_from_file(str(wav_path))
            fv = mfcc_vec.reshape(1, -1)
            fv_scaled = self.audio_mfcc_scaler.transform(fv)
            p_mfcc = 1 - self.audio_mfcc_model.predict(fv_scaled)[0, 0]  # Invert: model outputs prob_healthy
            probs.append(("mfcc", p_mfcc))
            parts["mfcc_prob_pd"] = float(p_mfcc)
        else:
            weights = (0.0, 1.0)  # fall back to spec only

        # Spectrogram CNN
        if self.audio_cnn is not None:
            spec = self._mel_spectrogram_image(str(wav_path))
            img = spec[..., np.newaxis]
            batch = np.expand_dims(img, 0)
            p_spec = 1 - self.audio_cnn.predict(batch)[0, 0]  # Invert: model outputs prob_healthy
            probs.append(("spec", p_spec))
            parts["spec_prob_pd"] = float(p_spec)

        # combine
        w_mfcc, w_spec = weights
        p_final = 0.0
        if "mfcc" in dict(probs):
            p_final += w_mfcc * dict(probs)["mfcc"]
        if "spec" in dict(probs):
            p_final += w_spec * dict(probs)["spec"]

        if p_final >= 0.5:
            label = "Parkinson"
            conf = p_final
        else:
            label = "Healthy"
            conf = 1.0 - p_final

        parts["final_label"] = label
        parts["final_prob_pd"] = float(p_final)
        parts["final_confidence"] = float(conf)
        return parts


voice_model = VoiceModel()
