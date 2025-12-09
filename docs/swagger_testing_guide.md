
# Swagger Testing Guide – Parkinson’s Disease Screening API

This guide explains how to **test all APIs step-by-step using Swagger UI**.

Swagger UI is automatically provided by FastAPI and allows interactive testing from the browser.

---

## 1️⃣ Start the API Server

Open terminal in your project root:

```

d:\OM\Parkisons_disease

````

Activate virtual environment:

```bash
.venv\Scripts\activate
````

Start the server:

```bash
uvicorn app.main:app --reload
```

If successful, you will see:

```text
Uvicorn running on http://127.0.0.1:8000
```

---

## 2️⃣ Open Swagger UI in Browser

Open this URL in your browser:

```
http://127.0.0.1:8000/docs
```

You will see the following APIs:

* `GET /`
* `POST /predict/drawing`
* `POST /predict/voice/audio`
* `POST /predict/voice/csv`
* `POST /predict/voice/ensemble`
* `POST /predict/voice/final-audio`

---

# ✅ API TESTING PROCEDURES

---

## ✅ 1. Test API Health – `GET /`

### Steps

1. Click `GET /`
2. Click **Try it out**
3. Click **Execute**

### Expected Response

```json
{
  "message": "Parkinson's Screening API is running"
}
```

---

## ✅ 2. Test Drawing Model – `POST /predict/drawing`

### Test Files

Use any of these:

* `spiral_healthy.png`
* `spiral_parkinson.png`
* `wave_healthy.png`
* `wave_parkinson.png`

### Steps

1. Click `POST /predict/drawing`
2. Click **Try it out**
3. Click **Choose File**
4. Select any image file (example: `spiral_parkinson.png`)
5. Click **Execute**

### Expected Response Format

```json
{
  "model": "drawing_hog_svm",
  "predicted_label": "Parkinson",
  "prob_pd_raw": 0.87,
  "confidence": 0.87
}
```

✅ If spiral_healthy → prediction should be **Healthy**
✅ If spiral_parkinson → prediction should be **Parkinson**

---

## ✅ 3. Test Audio CNN Model – `POST /predict/voice/audio`

### Test Files

* `voice_healthy.wav`
* `voice_parkinson.wav`

### Steps

1. Click `POST /predict/voice/audio`
2. Click **Try it out**
3. Click **Choose File**
4. Upload `.wav` file
5. Click **Execute**

### Expected Response Format

```json
{
  "model": "voice_audio_cnn",
  "predicted_label": "Healthy",
  "prob_pd_raw": 0.32,
  "confidence": 0.68
}
```

⚠️ This model is **experimental** and may often predict Healthy due to small dataset.

---

## ✅ 4. Test Voice CSV Model – `POST /predict/voice/csv`

This is your **strongest voice model**.

### Required Input

A **JSON object with all speech feature values**
(Columns must match `model_voice_csv_columns.joblib`)

### Example Test JSON (Partial – Not Complete)

```json
{
  "MDVP:Fo(Hz)": 119.992,
  "MDVP:Fhi(Hz)": 157.302,
  "MDVP:Jitter(%)": 0.00784,
  "MDVP:Shimmer": 0.04374,
  "DFA": 0.815,
  "RPDE": 0.469
}
```

⚠️ Full testing requires **all columns** from the CSV dataset.

### Steps

1. Click `POST /predict/voice/csv`
2. Click **Try it out**
3. Paste full JSON feature set
4. Click **Execute**

### Expected Response

```json
{
  "model": "voice_csv_mlp",
  "predicted_label": "Parkinson",
  "prob_pd_raw": 0.91,
  "confidence": 0.91
}
```

✅ This is the **production-ready voice classifier**.

---

## ✅ 5. Test Voice Ensemble – `POST /predict/voice/ensemble`

### Input

* CSV JSON Features
* `.wav` audio file

### Output

```json
{
  "csv_model": {...},
  "audio_model": {...},
  "final_label": "Parkinson",
  "final_prob_pd": 0.86,
  "final_confidence": 0.86
}
```

⚠️ This endpoint is **research-only**.

---

## ✅ 6. Test Final Audio Ensemble – `POST /predict/voice/final-audio`

This endpoint only needs a `.wav` file and combines:

* MFCC + MLP
* Spectrogram + CNN

### Steps

1. Click `POST /predict/voice/final-audio`
2. Click **Try it out**
3. Upload `.wav`
4. Click **Execute**

### Example Output

```json
{
  "model": "voice_audio_final_ensemble",
  "mfcc_prob_pd": 0.47,
  "spec_prob_pd": 0.42,
  "final_label": "Healthy",
  "final_prob_pd": 0.445,
  "final_confidence": 0.555
}
```

⚠️ This is **audio-only experimental prediction**.

---

# ✅ COMMON ERRORS & FIXES

| Error                   | Cause            | Fix                          |
| ----------------------- | ---------------- | ---------------------------- |
| 400 Invalid file        | Wrong format     | Upload `.png` or `.wav` only |
| 500 Internal error      | Model not loaded | Restart API                  |
| Missing feature         | JSON mismatch    | Use exact CSV column names   |
| Always predicts Healthy | Weak audio data  | Use CSV model instead        |

---

# ✅ FINAL TESTING STATUS

| API               | Status           |
| ----------------- | ---------------- |
| Drawing API       | ✅ Fully Working  |
| Voice CSV API     | ✅ Fully Working  |
| Voice Audio CNN   | ⚠️ Weak          |
| Voice Final Audio | ⚠️ Experimental  |
| Voice Ensemble    | ⚠️ Research Only |

---

# ✅ Testing Completed

All major pipelines are confirmed operational through Swagger.

This project is now:

* ✅ API Ready
* ✅ UI Ready
* ✅ Deployment Ready
* ✅ Report Ready

---



