import litellm
import pandas as pd

litellm.telemetry = False

PROVIDERS = {
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "model": "gemini/gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "openai": {
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}

SYSTEM_PROMPT = """Eres un asistente experto en análisis de datos con pandas.
El usuario te hará preguntas en español sobre un DataFrame llamado `df`.

Tu tarea es responder ÚNICAMENTE con código Python válido que:
1. Use solo pandas (ya importado como `pd`) y matplotlib (ya importado como `plt`).
2. Asigne el resultado final a una variable llamada `result`, O genere un gráfico con matplotlib (plt.show() NO, simplemente crea la figura).
3. No importe librerías adicionales.
4. Sea conciso y correcto.

Responde SOLO con el bloque de código, sin explicaciones, sin markdown, sin ```python.
"""


def build_schema_context(df: pd.DataFrame) -> str:
    cols = "\n".join(f"  - {col} ({df[col].dtype})" for col in df.columns)
    sample = df.head(3).to_string(index=False)
    return f"Columnas del DataFrame:\n{cols}\n\nPrimeras 3 filas:\n{sample}"


def ask_llm(df: pd.DataFrame, question: str, provider: str, api_key: str, error_feedback: str | None = None) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"Proveedor no soportado: {provider}")

    model = PROVIDERS[provider]["model"]
    schema = build_schema_context(df)

    user_content = f"{schema}\n\nPregunta del usuario: {question}"
    if error_feedback:
        user_content += f"\n\nEl código anterior falló con este error:\n{error_feedback}\nCorrige el código."

    import os
    os.environ[PROVIDERS[provider]["env_key"]] = api_key

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
    )
    code = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model includes them
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])

    return code.strip()
