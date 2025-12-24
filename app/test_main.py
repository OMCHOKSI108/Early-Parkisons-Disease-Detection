# app/test_main.py

import os
import pytest

# Set dummy DATABASE_URL for tests
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

def test_app_import():
    """Test that the app can be imported without errors."""
    from .main import app
    assert app is not None
    assert app.title == "Parkinson's Disease Screening API"

def test_database_import():
    """Test that database models can be imported."""
    from .database import User, History
    assert User is not None
    assert History is not None

# Add more tests as needed, e.g., model loading, but for now basic imports