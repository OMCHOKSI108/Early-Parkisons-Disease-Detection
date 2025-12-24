import logging
import os
from pathlib import Path
from . import config

logger = logging.getLogger(__name__)


def ensure_models_present():
    """Ensure model files exist locally. If `MODEL_S3_BUCKET` is set, attempt to download missing files from S3."""
    models_dir = Path(config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    bucket = config.MODEL_S3_BUCKET
    prefix = config.MODEL_S3_PREFIX or ""

    missing = []
    for fname in config.EXPECTED_MODEL_FILES:
        fpath = models_dir / fname
        if not fpath.exists():
            missing.append(fname)

    if not missing:
        logger.info("All model files present in %s", models_dir)
        return

    if not bucket:
        logger.warning("Missing model files but MODEL_S3_BUCKET not set. Missing: %s", missing)
        return

    try:
        import boto3
        s3 = boto3.client("s3")
    except Exception as e:
        logger.error("boto3 is required to download models from S3: %s", e)
        return

    for fname in missing:
        key = f"{prefix}/{fname}".lstrip("/") if prefix else fname
        dest = models_dir / fname
        try:
            logger.info("Downloading model s3://%s/%s -> %s", bucket, key, dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            logger.info("Downloaded %s", fname)
        except Exception as e:
            logger.error("Failed to download %s from S3: %s", key, e)
