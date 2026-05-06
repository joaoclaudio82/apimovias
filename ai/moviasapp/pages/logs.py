"""Logs – detalhes de execuções do pipeline."""

import pandas as pd
import streamlit as st

import api_client as api

st.title("📋 Logs de Pipeline")

tab_seg, tab_train, tab_ingest = st.tabs(["Segmentação", "Treino", "Ingestão"])

# ── Helper ────────────────────────────────────────────────────


def _show_runs(step: str, limit: int = 50):
    """Exibe execuções do pipeline para uma etapa."""
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Target", ["Todos", "km", "h", "global"], key=f"log_{step}_target")
    with col2:
        status = st.selectbox("Status", ["Todos", "completed", "failed", "running", "pending"], key=f"log_{step}_status")

    target_val = None if target == "Todos" else target
    status_val = None if status == "Todos" else status

    try:
        runs = api.list_pipeline_runs(step=step, target=target_val, status=status_val, limit=limit)
    except Exception as e:
        st.error(f"Erro: {e}")
        return

    if not runs:
        st.info(f"Nenhum log de {step}.")
        return

    df = pd.DataFrame(runs)
    status_emoji = {"pending": "🟡", "running": "🔵", "completed": "🟢", "failed": "🔴"}
    df["status_badge"] = df["status"].map(lambda s: f"{status_emoji.get(s, '⚪')} {s}")

    if "started_at" in df.columns and "finished_at" in df.columns:
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        df["finished_at"] = pd.to_datetime(df["finished_at"], errors="coerce")
        df["duração"] = (df["finished_at"] - df["started_at"]).apply(
            lambda d: str(d).split(".")[0] if pd.notna(d) else "—"
        )

    cols = ["id", "target", "model_type", "status_badge", "started_at", "duração"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(
        df[cols].rename(columns={
            "id": "ID", "target": "Target", "model_type": "Modelo",
            "status_badge": "Status", "started_at": "Início", "duração": "Duração",
        }),
        use_container_width=True, hide_index=True,
    )

    # Expand details
    for _, run in df.iterrows():
        with st.expander(f"Run #{run['id']} — {run['status']}"):
            if run.get("error_message"):
                st.error(run["error_message"])
            if run.get("metrics"):
                st.subheader("Métricas")
                st.json(run["metrics"])
            if run.get("artifacts"):
                st.subheader("Artefatos")
                st.json(run["artifacts"])


# ── Tabs ──────────────────────────────────────────────────────

with tab_seg:
    st.subheader("Segmentação")
    _show_runs("segmentation")

with tab_train:
    st.subheader("Treino")
    st.markdown("Inclui etapas: **profiles**, **datasets**, **optimization**")
    step_choice = st.selectbox(
        "Etapa específica",
        ["Todas", "profiles", "datasets", "optimization"],
        key="train_step",
    )
    if step_choice == "Todas":
        for s in ["profiles", "datasets", "optimization"]:
            st.markdown(f"---\n#### {s.title()}")
            _show_runs(s, limit=20)
    else:
        _show_runs(step_choice)

with tab_ingest:
    st.subheader("Ingestão")
    _show_runs("ingestion")
