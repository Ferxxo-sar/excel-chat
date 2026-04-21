import os
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

CODE_PROMPT = """Sos un experto en pandas. Respondé SOLO con código Python válido, sin explicaciones, sin markdown, sin ```.

Tenés acceso a:
- `df`: el DataFrame con los datos
- `pd`: pandas
- `plt`: matplotlib.pyplot

Asigná el resultado a `result`, o generá un gráfico con matplotlib (sin plt.show()).
No hagas imports adicionales."""

EXPLAIN_PROMPT = """Respondé en español, en 1 oración clara y directa, como un empleado que conoce los números.
No menciones código ni pandas. Si hay un número, destacalo. Si hay varios, mencioná el más importante."""


def build_schema_context(df: pd.DataFrame) -> str:
    cols = "\n".join(f"  - {col} ({df[col].dtype})" for col in df.columns)
    sample = df.head(5).to_string(index=False)
    return f"Columnas:\n{cols}\n\nDatos (primeras filas):\n{sample}"


def _call(provider: str, api_key: str, system: str, user: str, max_tokens: int = 1024) -> str:
    os.environ[PROVIDERS[provider]["env_key"]] = api_key
    response = litellm.completion(
        model=PROVIDERS[provider]["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
    return text.strip()


def generate_code(df: pd.DataFrame, question: str, provider: str, api_key: str, error_feedback: str | None = None) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"Proveedor no soportado: {provider}")
    schema = build_schema_context(df)
    user = f"{schema}\n\nPregunta: {question}"
    if error_feedback:
        user += f"\n\nEl código anterior falló:\n{error_feedback}\nCorregilo."
    return _call(provider, api_key, CODE_PROMPT, user)


def explain_result(question: str, result_str: str, provider: str, api_key: str) -> str:
    user = f'Pregunta: "{question}"\nResultado: {result_str}'
    try:
        return _call(provider, api_key, EXPLAIN_PROMPT, user, max_tokens=150)
    except Exception:
        return result_str
