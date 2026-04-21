import anthropic
import pandas as pd

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
    cols = "\n".join(
        f"  - {col} ({df[col].dtype})" for col in df.columns
    )
    sample = df.head(3).to_string(index=False)
    return f"Columnas del DataFrame:\n{cols}\n\nPrimeras 3 filas:\n{sample}"


def ask_claude(df: pd.DataFrame, question: str, api_key: str, error_feedback: str | None = None) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    schema = build_schema_context(df)

    user_content = f"{schema}\n\nPregunta del usuario: {question}"
    if error_feedback:
        user_content += f"\n\nEl código anterior falló con este error:\n{error_feedback}\nCorrige el código."

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return message.content[0].text.strip()
