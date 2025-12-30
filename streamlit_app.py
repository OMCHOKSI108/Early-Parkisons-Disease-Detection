import streamlit as st
import json
from pathlib import Path
from io import BytesIO

import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog

from app import config

# ---------------- UTILS ----------------

@st.cache_resource(show_spinner=False)
def load_drawing_models():
    scaler_path = Path(config.HOG_SCALER_PATH)
    svm_path = Path(config.HOG_SVM_PATH)

    if not scaler_path.exists() or not svm_path.exists():
        return None, None

    return joblib.load(scaler_path), joblib.load(svm_path)


def preprocess_image(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("L")
    img = img.resize((256, 256))
    arr = np.array(img) / 255.0

    feat = hog(
        arr,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return feat.reshape(1, -1)


def predict_drawing(img_bytes):
    scaler, svm = load_drawing_models()
    if scaler is None:
        raise RuntimeError("Drawing model files missing")

    feat = preprocess_image(img_bytes)
    feat_s = scaler.transform(feat)

    prob = svm.predict_proba(feat_s)[0, 1]
    label = "Parkinson" if prob >= 0.5 else "Healthy"
    conf = prob if prob >= 0.5 else 1 - prob

    return {
        "predicted_label": label,
        "prob_pd_raw": float(prob),
        "confidence": float(conf)
    }


@st.cache_resource(show_spinner=False)
def load_csv_model():
    """Load voice CSV model, scaler, and columns once per process."""
    # Lazy import TensorFlow so drawing-only usage doesn't pay the cost
    import tensorflow as tf

    m = Path(config.VOICE_CSV_MODEL_PATH)
    s = Path(config.VOICE_CSV_SCALER_PATH)
    c = Path(config.VOICE_CSV_COLUMNS_PATH)

    if not (m.exists() and s.exists() and c.exists()):
        return None, None, None

    return (
        tf.keras.models.load_model(m),
        joblib.load(s),
        joblib.load(c)
    )


def predict_csv_features(feats):
    model, scaler, cols = load_csv_model()
    if model is None:
        raise RuntimeError("CSV voice model missing")

    vec = [float(feats[c]) for c in cols]
    arr = np.array(vec).reshape(1, -1)
    arr_s = scaler.transform(arr)

    prob = float(model.predict(arr_s)[0][0])
    label = "Parkinson" if prob >= 0.5 else "Healthy"
    conf = prob if prob >= 0.5 else 1 - prob

    return {
        "predicted_label": label,
        "prob_pd_raw": prob,
        "confidence": float(conf)
    }

# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Parkinson Screening", layout="wide")

CUSTOM_CSS = """
<style>
  .stApp {
    background-color: #f9fafb;
  }
  .main-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
  }
  .subtitle {
    color: #6b7280;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }
  .card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(15,23,42,0.06);
  }
  .result-card {
    margin-top: 0.5rem;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
  }
  .result-label-healthy {
    color: #16a34a;
    font-weight: 600;
    font-size: 1.1rem;
  }
  .result-label-parkinson {
    color: #dc2626;
    font-weight: 600;
    font-size: 1.1rem;
  }
  .result-metrics {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.35rem;
    font-size: 0.9rem;
    color: #4b5563;
  }
  .result-metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #9ca3af;
  }
  .stButton > button {
    background-color: #0f766e;
    color: #ffffff;
    border-radius: 999px;
    border: none;
    padding: 0.45rem 1.2rem;
    font-weight: 600;
  }
  .stButton > button:hover {
    background-color: #115e59;
  }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("<div class='main-title'>Parkinson's Disease Screening</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Experimental screening UI for local use. Not a clinical diagnostic tool.</div>", unsafe_allow_html=True)


def render_result_card(res: dict, title: str):
    label = res.get("predicted_label", "-")
    conf = res.get("confidence", 0.0)
    prob = res.get("prob_pd_raw", 0.0)
    cls = "result-label-healthy" if label == "Healthy" else "result-label-parkinson"

    st.markdown("""<div class='result-card'>""", unsafe_allow_html=True)
    st.markdown(f"<div class='{cls}'>{label}</div>", unsafe_allow_html=True)
    st.markdown("<div class='result-metrics'>" """  <div>
        <div class='result-metric-label'>Confidence</div>
        <div>{:.3f}</div>
      </div>
      <div>
        <div class='result-metric-label'>PD Probability</div>
        <div>{:.3f}</div>
      </div>
    </div>""".format(conf, prob), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


tab_drawing, tab_voice = st.tabs(["Drawing (Spiral / Wave)", "Voice (CSV Features)"])

with tab_drawing:
    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Drawing Input")
        st.write("Upload a spiral or wave drawing (PNG / JPG). The model analyzes tremor patterns using HOG + SVM.")

        img_file = st.file_uploader(
            "Upload spiral / wave image",
            type=["png", "jpg", "jpeg"],
            key="drawing-upload"
        )

        img_bytes = None
        if img_file is not None:
            img_bytes = img_file.read()
            try:
                st.image(Image.open(BytesIO(img_bytes)), width=520)
            except Exception:
                st.info("Image preview not available, but file is loaded.")

        run = st.button("Run Drawing Prediction")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Result")
        if run:
            if img_bytes is None:
                st.warning("Please upload an image first.")
            else:
                try:
                    with st.spinner("Running drawing model..."):
                        res = predict_drawing(img_bytes)
                    render_result_card(res, "Drawing Prediction")
                    with st.expander("Raw output"):
                        st.json(res)
                except Exception as e:
                    st.error(str(e))
        else:
            st.info("Upload an image and click 'Run Drawing Prediction' to see results.")
        st.markdown("</div>", unsafe_allow_html=True)


with tab_voice:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Voice Features JSON")
        st.write("Paste a JSON object with all required acoustic features matching the training columns.")

        txt = st.text_area("Feature JSON", height=260, key="csv-json")
        run_csv = st.button("Run CSV Prediction")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Result")
        if run_csv:
            if not txt.strip():
                st.warning("Please paste feature JSON first.")
            else:
                try:
                    feats = json.loads(txt)
                    with st.spinner("Running voice CSV model..."):
                        res = predict_csv_features(feats)
                    render_result_card(res, "Voice CSV Prediction")
                    with st.expander("Raw output"):
                        st.json(res)
                except Exception as e:
                    st.error(str(e))
        else:
            st.info("Paste JSON features and click 'Run CSV Prediction' to see results.")
        st.markdown("</div>", unsafe_allow_html=True)

st.caption("Local Streamlit UI for experimentation. For production use the FastAPI service.")
