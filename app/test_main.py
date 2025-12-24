# # app/test_main.py

# import os
# import pytest

# # Set dummy DATABASE_URL for tests
# os.environ["DATABASE_URL"] = "sqlite:///./test.db"

# def test_app_import():
#     """Test that the app can be imported without errors."""
#     from app.main import app
#     assert app is not None
#     assert app.title == "Parkinson's Disease Screening API"

# def test_database_import():
#     """Test that database models can be imported."""
#     from app.database import User, History
#     assert User is not None
#     assert History is not None

# # Add more tests as needed, e.g., model loading, but for now basic imports

from pathlib import Path
from app.drawing_model import drawing_model

p = Path("path/to/wallpaper.jpg")
with open(p, "rb") as f:
    res = drawing_model.predict_from_bytes(f.read())
print(res)