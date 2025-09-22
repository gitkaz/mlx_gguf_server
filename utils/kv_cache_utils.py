from safetensors import safe_open
from fastapi import HTTPException, UploadFile
import os
from pathlib import Path
import gzip
import tempfile

from logging import getLogger

logger = getLogger("uvicorn")

def get_kv_cache_abs_path():
    """
    return absolute path of kv_cache directory.
    """
    project_root_abs_path = Path(__file__).parent.parent
    kv_cache_abs_path = project_root_abs_path.joinpath('worker', 'kv_cache', 'data')
    return str(kv_cache_abs_path)


def prepare_temp_dir() -> Path:
    tmp_base_dir = Path(os.environ["KV_CACHE_PATH"]).joinpath('tmp')
    os.makedirs(tmp_base_dir, exist_ok=True)  # Ensure the tmp base directory exists
    temp_dir = tempfile.mkdtemp(dir=tmp_base_dir)
    return Path(temp_dir)

def validate_filename(filename: str) -> None:
    """Ensure filename matches expected patterns."""
    suffixes = (".safetensors", ".safetensors.gz")
    if not filename.endswith(suffixes):
        raise HTTPException(400, f"Filename must be one of: {', '.join(suffixes)}")

def __validate_safetensors(file_path: Path) -> bool:
    """Check if a file is a valid safetensors file."""
    try:
        with safe_open(file_path, framework="pt") as f:
            # Just check it can be opened
            return True
    except Exception:
        raise HTTPException(400, "Invalid safetensors file format")

async def process_upload_file(file: UploadFile, file_path: Path) -> Path:
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Decompress if necessary
    decompressed_path = file_path
    if file_path.suffix == '.safetensors':
        pass
    elif file_path.suffix == '.gz':
        decompressed_path = Path(file_path.stem)
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())
        file_path.unlink(missing_ok=True) # Remove compressed file

    # Validate safetensors format
    __validate_safetensors(decompressed_path)

    return decompressed_path