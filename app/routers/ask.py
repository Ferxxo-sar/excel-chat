from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
from app.storage import load_dataframe
from app.llm_client import generate_code, explain_result, PROVIDERS
from app.code_runner import run_code

router = APIRouter()


class AskRequest(BaseModel):
    file_id: str
    question: str
    api_key: str
    provider: str = "groq"


@router.get("/providers")
def list_providers():
    return {"providers": list(PROVIDERS.keys())}


def _result_to_str(result: dict) -> str:
    if result["type"] == "text":
        return str(result["data"])
    if result["type"] == "table":
        rows = result["data"]
        if not rows:
            return "Sin resultados."
        # Show up to 10 rows as compact text
        lines = []
        for row in rows[:10]:
            lines.append(", ".join(f"{k}: {v}" for k, v in row.items() if v is not None))
        return "\n".join(lines)
    return ""


@router.post("/ask")
def ask(body: AskRequest):
    if not body.api_key:
        raise HTTPException(400, "API key requerida.")
    if body.provider not in PROVIDERS:
        raise HTTPException(400, f"Proveedor no soportado. Opciones: {list(PROVIDERS.keys())}")

    try:
        df = load_dataframe(body.file_id)
    except KeyError:
        raise HTTPException(404, "Archivo no encontrado. Volvé a subirlo.")

    # Step 1: generate code
    try:
        code = generate_code(df, body.question, body.provider, body.api_key)
    except Exception as e:
        raise HTTPException(502, f"Error al contactar el LLM: {e}")

    # Step 2: execute code (with 1 retry)
    try:
        result = run_code(code, df)
    except Exception as first_error:
        try:
            code = generate_code(df, body.question, body.provider, body.api_key, error_feedback=str(first_error))
            result = run_code(code, df)
        except Exception as second_error:
            raise HTTPException(422, f"No se pudo ejecutar el análisis: {second_error}")

    # Step 3: explain the actual result in natural language
    if result["type"] != "chart":
        result_str = _result_to_str(result)
        result["explanation"] = explain_result(body.question, result_str, body.provider, body.api_key)

    return result
