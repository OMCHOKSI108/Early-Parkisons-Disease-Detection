# Project Structure

This file documents the top-level project layout and brief descriptions of key folders/files.

- Dockerfile: Docker build instructions.
- README.md: Project overview and usage.
- render.yaml / runtime.txt: Deployment/runtime configs.

- app/
  - __init__.py: Flask app package initializer.
  - main.py: Application routes and entrypoints.
  - model_loader.py: Loads ML models from `models/`.
  - drawing_model.py: Spiral-drawing model utilities.
  - voice_model.py: Voice-model utilities.
  - database.py: Database helpers.
  - config.py: Configuration values.
  - requirements.txt: Python dependencies for the app.
  - static/: Static assets such as styles.css.
  - templates/: HTML templates (index.html).
  - test_main.py: App-level tests.

- models/
  - spiral/: Trained spiral models and scalers (e.g. `.keras`, `.joblib`).
  - voice/: Trained voice models and scalers.

- dataset/
  - spiral/: Spiral drawing dataset.
  - voice/: Voice dataset samples.

- ml/: Local virtualenv and vendor packages used during model development.

- assets/: Misc project assets.
- pngs/: Stored PNG images used for documentation or examples.

- docs/
  - README_DEPLOY_AWS.md: AWS deployment notes.
  - SCREENSHOTS.md: Screenshots and examples.
  - swagger_testing_guide.md: API testing guide.
  - TECHNICAL_SUMMARY.md: High-level architecture notes.
  - strcture.md: (this file) project structure summary.

- noteboooks/: Jupyter notebooks for experimentation (typo preserved from repo).

- test/: Additional tests.

Notes:
- Trained model files live under `models/` and are large; they are not committed to source control in many projects but are present here for convenience.
- The `app/` folder contains the Flask application and is the primary runtime code.

If you want, I can also:
- Add this structure into `README.md`.
- Generate a visual tree file (text or image).
