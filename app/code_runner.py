import re
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORBIDDEN = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+socket\b",
    r"\b__import__\b",
    r"\bopen\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\burllib\b",
    r"\brequests\b",
    r"\bhttpx\b",
    r"\bshutil\b",
    r"\bos\.",
    r"\bsys\.",
]


def validate_code(code: str) -> None:
    for pattern in FORBIDDEN:
        if re.search(pattern, code):
            raise ValueError(f"Código no permitido: patrón '{pattern}' detectado")


def run_code(code: str, df: pd.DataFrame) -> dict:
    validate_code(code)

    local_scope = {"df": df, "pd": pd, "plt": plt}
    exec(compile(code, "<generated>", "exec"), {"__builtins__": {}}, local_scope)

    # Check if a figure was created
    fig = plt.gcf()
    if fig.get_axes():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        image_b64 = base64.b64encode(buf.getvalue()).decode()
        return {"type": "chart", "image": image_b64}

    plt.close("all")

    result = local_scope.get("result")
    if result is None:
        raise ValueError("El código no asignó ningún valor a la variable 'result'")

    if isinstance(result, pd.DataFrame):
        return {"type": "table", "data": result.where(pd.notnull(result), None).to_dict(orient="records")}

    if isinstance(result, pd.Series):
        return {"type": "table", "data": result.reset_index().where(pd.notnull(result.reset_index()), None).to_dict(orient="records")}

    return {"type": "text", "data": str(result)}
