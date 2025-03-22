from safetensors.torch import safe_open
from fastapi import HTTPException, UploadFile
import re
import os
import gzip
import tempfile

from logging import getLogger

logger = getLogger("uvicorn")

def prepare_temp_dir(tmp_base_dir: str) -> str:
    os.makedirs(tmp_base_dir, exist_ok=True)  # Ensure the tmp base directory exists
    temp_dir = tempfile.mkdtemp(dir=tmp_base_dir)
    return temp_dir

def validate_session_id(session_id: str) -> None:
    """Validate session ID is a valid UUIDv4."""
    if not re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', session_id):
        raise HTTPException(400, "Invalid session ID format (must be UUIDv4)")

def validate_filename(session_id: str, filename: str) -> None:
    """Ensure filename matches expected patterns."""
    expected = [f"{session_id}.safetensors", f"{session_id}.safetensors.gz"]
    if filename not in expected:
        raise HTTPException(400, f"Filename must be one of: {', '.join(expected)}")

def __validate_safetensors(file_path: str) -> bool:
    """Check if a file is a valid safetensors file."""
    try:
        with safe_open(file_path, framework="pt") as f:
            # Just check it can be opened
            return True
    except Exception:
        raise HTTPException(400, "Invalid safetensors file format")

async def process_upload_file(file: UploadFile, file_path: str) -> str:
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    _, ext = os.path.splitext(file.filename)

    # Decompress if necessary
    if ext == '.safetensors':
        decompressed_path = file_path
    elif ext == '.gz':
        decompressed_path = os.path.splitext(file_path)[0]
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(file_path)  # Remove compressed file

    # Validate safetensors format
    __validate_safetensors(decompressed_path)

    return decompressed_path