from pathlib import Path
from typing import Tuple
import uuid

from ..settings import settings

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_upload(file_name: str, data: bytes) -> str:
    ext = Path(file_name).suffix
    unique = f"{uuid.uuid4()}{ext}"
    dest = UPLOAD_DIR / unique
    dest.write_bytes(data)
    return str(dest)