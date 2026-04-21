import os
import json
import litellm
import pandas as pd

litellm.telemetry = False

PROVIDERS = {
    "groq": {
        "model": "groq/llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
    },
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "model": "gemini/gemini-1.5-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}

SYSTEM_PROMPT = """Sos un analista de datos experto que responde preguntas en español sobre planillas de Excel o CSV.

Cuando el usuario haga una pregunta, respondé con un JSON válido con este formato exacto:
{
  "explanation": "Respuesta en lenguaje natural, como si fueras un contador o empleado que conoce los números. 1-3 oraciones, directas y útiles. Destacá los números clave.",
  "code": "código Python que usa el DataFrame `df` (pandas importado como `pd`, matplotlib como `plt`). Asigná el resultado a `result` o generá un gráfico. Sin imports adicionales."
}

Reglas:
- El JSON debe ser válido, sin texto antes ni después.
- El campo `explanation` es la respuesta humana, sin mencionar pandas ni código.
- Si la pregunta pide un gráfico, en `explanation` describí qué muestra el gráfico.
- Si el código falla, el sistema te va a avisar para que lo corrijas."""


def build_schema_context(df: pd.DataFrame) -> str:
    cols = "\n".join(f"  - {col} ({df[col].dtype})" for col in df.columns)
    sample = df.head(5).to_string(index=False)
    return f"Columnas disponibles:\n{cols}\n\nPrimeras filas de datos:\n{sample}"


def _call_llm(provider: str, api_key: str, messages: list) -> str:
    model = PROVIDERS[provider]["model"]
    os.environ[PROVIDERS[provider]["env_key"]] = api_key
    response = litellm.completion(model=model, messages=messages, max_tokens=1500)
    return response.choices[0].message.content.strip()


def _parse_response(text: str) -> dict:
    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```")[1] if "```json" not in text else text.split("```json")[1].split("```")[0]
    text = text.strip()
    return json.loads(text)


def ask_llm(df: pd.DataFrame, question: str, provider: str, api_key: str, error_feedback: str | None = None) -> dict:
    if provider not in PROVIDERS:
        raise ValueError(f"Proveedor no soportado: {provider}")

    schema = build_schema_context(df)
    user_content = f"{schema}\n\nPregunta: {question}"
    if error_feedback:
        user_content += f"\n\nEl código anterior falló con este error:\n{error_feedback}\nCorregí solo el campo `code`, manteniendo el mismo `explanation`."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = _call_llm(provider, api_key, messages)
    return _parse_response(raw)
