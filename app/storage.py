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


def _find_header_row(path: Path, is_csv: bool) -> int:
    """Find the row index that contains the real column headers."""
    if is_csv:
        raw = pd.read_csv(path, header=None, nrows=25, dtype=str)
    else:
        raw = pd.read_excel(path, header=None, nrows=25, dtype=str)

    best_row = 0
    best_score = -1

    for i, row in raw.iterrows():
        values = row.dropna().tolist()
        values = [str(v).strip() for v in values if str(v).strip()]
        if len(values) < 2:
            continue
        # Score: number of short string values (likely column names)
        score = sum(1 for v in values if len(v) <= 50 and not v.replace(".", "").replace(",", "").replace("-", "").isdigit())
        if score > best_score:
            best_score = score
            best_row = i

    return best_row


def load_dataframe(file_id: str) -> pd.DataFrame:
    path = _store.get(file_id)
    if path is None or not path.exists():
        raise KeyError(f"file_id {file_id!r} not found")

    is_csv = path.suffix == ".csv"
    header_row = _find_header_row(path, is_csv)

    if is_csv:
        df = pd.read_csv(path, skiprows=header_row, header=0)
    else:
        df = pd.read_excel(path, header=header_row)

    # Drop fully empty columns and rows
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all").reset_index(drop=True)

    # Remove "Unnamed: X" columns that sneak through
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed:")]]

    return df
