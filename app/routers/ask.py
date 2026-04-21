from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.storage import load_dataframe
from app.claude_client import ask_claude
from app.code_runner import run_code
import anthropic

router = APIRouter()


class AskRequest(BaseModel):
    file_id: str
    question: str
    api_key: str


@router.post("/ask")
def ask(body: AskRequest):
    if not body.api_key.startswith("sk-ant-"):
        raise HTTPException(400, "API key inválida. Debe empezar con 'sk-ant-'.")

    try:
        df = load_dataframe(body.file_id)
    except KeyError:
        raise HTTPException(404, "Archivo no encontrado. Volvé a subirlo.")

    try:
        code = ask_claude(df, body.question, body.api_key)
    except anthropic.AuthenticationError:
        raise HTTPException(401, "API key incorrecta o sin permisos.")
    except anthropic.APIError as e:
        raise HTTPException(502, f"Error al contactar la API de Anthropic: {e}")

    try:
        return run_code(code, df)
    except Exception as first_error:
        try:
            code = ask_claude(df, body.question, body.api_key, error_feedback=str(first_error))
            return run_code(code, df)
        except Exception as second_error:
            raise HTTPException(422, f"No se pudo responder la pregunta: {second_error}")
