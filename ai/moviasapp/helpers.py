"""Helpers e constantes compartilhados entre páginas."""

QUALITY_LABELS = {
    0: "✅ Válida",
    1: "⚠️ Outlier",
    2: "🚫 Não modelável",
    3: "❌ Vazia",
}

QUALITY_COLORS = {
    0: "green",
    1: "orange",
    2: "red",
    3: "gray",
}


import math


def quality_label(code: int | float | None) -> str:
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return QUALITY_LABELS[3]  # Vazia
    return QUALITY_LABELS.get(int(code), f"? ({code})")
