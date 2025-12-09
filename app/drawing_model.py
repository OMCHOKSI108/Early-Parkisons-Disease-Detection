# app/drawing_model.py

import io
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

from .config import HOG_SVM_PATH, HOG_SCALER_PATH


class DrawingModel:
    def __init__(self):
        self.svm = joblib.load(HOG_SVM_PATH)
        self.scaler = joblib.load(HOG_SCALER_PATH)

    @staticmethod
    def _preprocess_image(img: Image.Image) -> np.ndarray:
        """
        Convert PIL image -> HOG feature vector (same as in notebook).
        """
        # convert to grayscale numpy
        img = img.convert("L")           # grayscale
        img = img.resize((256, 256))     # same size as training

        img_arr = np.array(img) / 255.0

        # HOG parameters should match your notebook
        features = hog(
            img_arr,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        return features

    def predict_from_bytes(self, file_bytes: bytes) -> Dict:
        img = Image.open(io.BytesIO(file_bytes))
        feat = self._preprocess_image(img).reshape(1, -1)

        feat_scaled = self.scaler.transform(feat)
        prob_pd = self.svm.predict_proba(feat_scaled)[0, 1]  # prob of Parkinson

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


drawing_model = DrawingModel()
