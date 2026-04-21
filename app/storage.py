import uuid
import tempfile
from pathlib import Path
import pandas as pd

UPLOAD_DIR = Path(tempfile.gettempdir()) / "excel_chat"
UPLOAD_DIR.mkdir(exist_ok=True)

_store: dict[str, Path] = {}


def save_file(content: bytes, filename: str) -> str:
    file_id = str(uuid.uuid4())
    suffix = Path(filename).suffix.lower()
    path = UPLOAD_DIR / f"{file_id}{suffix}"
    path.write_bytes(content)
    _store[file_id] = path
    return file_id


def load_dataframe(file_id: str) -> pd.DataFrame:
    path = _store.get(file_id)
    if path is None or not path.exists():
        raise KeyError(f"file_id {file_id!r} not found")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)
