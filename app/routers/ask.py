from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.storage import load_dataframe
from app.llm_client import ask_llm, PROVIDERS
from app.code_runner import run_code

router = APIRouter()


class AskRequest(BaseModel):
    file_id: str
    question: str
    api_key: str
    provider: str = "anthropic"


@router.get("/providers")
def list_providers():
    return {"providers": list(PROVIDERS.keys())}


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

    try:
        code = ask_llm(df, body.question, body.provider, body.api_key)
    except Exception as e:
        raise HTTPException(502, f"Error al contactar el LLM: {e}")

    try:
        return run_code(code, df)
    except Exception as first_error:
        try:
            code = ask_llm(df, body.question, body.provider, body.api_key, error_feedback=str(first_error))
            return run_code(code, df)
        except Exception as second_error:
            raise HTTPException(422, f"No se pudo responder la pregunta: {second_error}")
