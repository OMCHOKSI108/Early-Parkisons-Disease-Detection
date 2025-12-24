# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Simple, small image for Render deployments
WORKDIR /app

# Install runtime deps needed for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better Docker layer caching
COPY app/requirements.txt /app/requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY app/ /app/

# Create a non-root user for better security
RUN groupadd -r appuser && useradd -r -g appuser appuser && chown -R appuser:appuser /app
USER appuser

# Expose application port
EXPOSE 8000

# Simple healthcheck for orchestrators
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s CMD curl -f http://127.0.0.1:${PORT:-8000}/health || exit 1

# Start the app. Use shell form so ${PORT} expands.
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"