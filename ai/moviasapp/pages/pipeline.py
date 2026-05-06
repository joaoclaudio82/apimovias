"""Monitoramento de execuções do pipeline."""

import pandas as pd
import streamlit as st

import api_client as api

st.title("⚙️ Pipeline / Tasks")

# ── Filters ───────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
with col1:
    step = st.selectbox("Etapa", ["Todos", "ingestion", "segmentation", "profiles", "datasets", "optimization"])
with col2:
    target_f = st.selectbox("Target", ["Todos", "km", "h", "global"])
with col3:
    status_f = st.selectbox("Status", ["Todos", "pending", "running", "completed", "failed"])

step_val = None if step == "Todos" else step
target_val = None if target_f == "Todos" else target_f
status_val = None if status_f == "Todos" else status_f

# ── Pagination ────────────────────────────────────────────────

limit = 25
page = st.number_input("Página", min_value=1, value=1, step=1)
offset = (page - 1) * limit

# ── Load ──────────────────────────────────────────────────────

try:
    runs = api.list_pipeline_runs(
        step=step_val, target=target_val, status=status_val,
        offset=offset, limit=limit,
    )
except Exception as e:
    st.error(f"Erro: {e}")
    st.stop()

if not runs:
    st.info("Nenhuma execução encontrada.")
    st.stop()

# ── Table ─────────────────────────────────────────────────────

df = pd.DataFrame(runs)

# Status badges
status_emoji = {"pending": "🟡", "running": "🔵", "completed": "🟢", "failed": "🔴"}
df["status_badge"] = df["status"].map(lambda s: f"{status_emoji.get(s, '⚪')} {s}")

# Duration
if "started_at" in df.columns and "finished_at" in df.columns:
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["finished_at"] = pd.to_datetime(df["finished_at"], errors="coerce")
    df["duração"] = (df["finished_at"] - df["started_at"]).apply(
        lambda d: str(d).split(".")[0] if pd.notna(d) else "—"
    )

display_cols = ["id", "step", "target", "model_type", "status_badge", "started_at", "duração"]
display_cols = [c for c in display_cols if c in df.columns]

st.dataframe(
    df[display_cols].rename(columns={
        "id": "ID", "step": "Etapa", "target": "Target",
        "model_type": "Modelo", "status_badge": "Status",
        "started_at": "Início", "duração": "Duração",
    }),
    use_container_width=True, hide_index=True,
)

# ── Detail expander ───────────────────────────────────────────

st.divider()
run_id = st.number_input("Detalhar ID da execução", min_value=1, step=1, value=int(df["id"].iloc[0]))

if st.button("Ver detalhes"):
    try:
        detail = api.get_pipeline_run(run_id)
    except Exception as e:
        st.error(f"Erro: {e}")
        st.stop()

    col_a, col_b = st.columns(2)
    col_a.metric("Etapa", detail["step"])
    col_b.metric("Status", detail["status"])

    if detail.get("error_message"):
        st.error(f"**Erro:** {detail['error_message']}")

    if detail.get("metrics"):
        st.subheader("Métricas")
        st.json(detail["metrics"])

    if detail.get("artifacts"):
        st.subheader("Artefatos")
        st.json(detail["artifacts"])
