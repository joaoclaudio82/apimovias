"""Home – visão geral do sistema."""

import streamlit as st
import api_client as api

st.title("🚛 Movias — Painel de Controle")

# ── Health ────────────────────────────────────────────────────

try:
    h = api.health()
    services = h.get("services", {})
    col1, col2 = st.columns(2)
    for name, info in services.items():
        c = col1 if name == "ai" else col2
        ok = info.get("status") == "ok"
        c.metric(name.upper(), "🟢 Online" if ok else "🔴 Offline")
except Exception as e:
    st.error(f"Gateway indisponível: {e}")
    st.stop()

# ── Profile metadata ─────────────────────────────────────────

st.divider()
st.subheader("Último perfil carregado")

try:
    meta = api.get_profile_metadata(last=True)
    if meta:
        m = meta[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Veículos", m["n_veiculos"])
        c2.metric("Início", m["dt_inicio"])
        c3.metric("Fim", m["dt_fim"])
        c4.metric("Tamanho da amostra", m["sample_size"])
    else:
        st.info("Nenhum perfil carregado. Execute a ingestão.")
except Exception as e:
    st.warning(f"Erro ao carregar metadata: {e}")

# ── Recent pipeline runs ─────────────────────────────────────

st.divider()
st.subheader("Execuções recentes")

try:
    runs = api.list_pipeline_runs(limit=5)
    if runs:
        import pandas as pd
        df = pd.DataFrame(runs)[["id", "step", "target", "status", "started_at", "finished_at"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma execução registrada.")
except Exception as e:
    st.warning(f"Erro: {e}")
