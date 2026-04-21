from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from app.storage import save_file, load_dataframe

router = APIRouter()

ALLOWED_EXTENSIONS = {".xlsx", ".csv"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Solo se aceptan archivos .xlsx y .csv")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "El archivo no puede superar 10 MB")

    file_id = save_file(content, file.filename)

    df = load_dataframe(file_id)

    preview_rows = df.head(5).where(pd.notnull(df), None).to_dict(orient="records")
    columns = [
        {"name": col, "type": str(df[col].dtype)}
        for col in df.columns
    ]

    return {
        "file_id": file_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": columns,
        "preview": preview_rows,
    }
