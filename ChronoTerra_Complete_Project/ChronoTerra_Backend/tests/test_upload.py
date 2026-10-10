import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app, MAX_UPLOAD_BYTES

client = TestClient(app)

def test_upload_below_limit():
    content = b"a" * (1024 * 1024 * 1) # Just 1MB for speed
    response = client.post("/api/inspect-image", files={"file": ("test.txt", content, "text/plain")})
    assert response.status_code == 422 # Fails inspection properly (Unsupported format)

def test_upload_above_limit():
    content = b"a" * (MAX_UPLOAD_BYTES + 1)
    response = client.post("/api/inspect-image", files={"file": ("test.txt", content, "text/plain")})
    assert response.status_code == 413 # Entity too large

def test_cleanup_after_failure():
    # If the upload is valid but the file fails inspection, the temp file should be deleted.
    import glob
    import tempfile
    
    initial_tmp_files = set(glob.glob(os.path.join(tempfile.gettempdir(), "chronoterra_upload_*.tmp")))
    
    # Trigger a request that fails validation
    response = client.post("/api/inspect-image", files={"file": ("test.txt", b"dummy", "text/plain")})
    assert response.status_code == 422
    
    final_tmp_files = set(glob.glob(os.path.join(tempfile.gettempdir(), "chronoterra_upload_*.tmp")))
    
    # New temp files should have been cleaned up
    assert final_tmp_files.issubset(initial_tmp_files)
