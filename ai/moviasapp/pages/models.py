"""Modelos – visualizar e trocar modelos ONNX ativos."""

import pandas as pd
import streamlit as st

import api_client as api

st.title("🧠 Modelos ONNX")

# ── Active models ─────────────────────────────────────────────

st.subheader("Modelos vigentes")

try:
    active = api.list_active_models()
    if active:
        df_active = pd.DataFrame(active)
        cols = ["target", "model_type", "version_id", "trained_at", "activated_at"]
        cols = [c for c in cols if c in df_active.columns]
        st.dataframe(df_active[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum modelo ativo registrado. Treine um modelo primeiro.")
except Exception as e:
    st.error(f"Erro: {e}")

# ── Swap model ────────────────────────────────────────────────

st.divider()
st.subheader("Trocar modelo ativo")

swap_target = st.selectbox("Target", ["km", "h"], key="swap_target")

try:
    available = api.list_available_models(swap_target)
    if available:
        # Construir opções para o select
        options = {
            f"{v['version_id']}  ({v['model_type']}, {v['created_at'][:10]})": v["version_id"]
            for v in available
        }
        selected_label = st.selectbox("Bundle disponível", list(options.keys()))
        selected_version = options[selected_label]

        if st.button("Ativar este modelo", type="primary"):
            try:
                result = api.set_active_model(swap_target, selected_version)
                st.success(f"Modelo ativo atualizado: **{result.get('version_id', '?')}**")
                st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")

        # Detalhes dos bundles disponíveis
        st.divider()
        st.subheader("Modelos disponíveis")
        df_av = pd.DataFrame(available)
        st.dataframe(df_av, use_container_width=True, hide_index=True)
    else:
        st.info(f"Nenhum bundle disponível para **{swap_target}**. Execute o treinamento primeiro.")
except Exception as e:
    st.error(f"Erro ao carregar modelos disponíveis: {e}")
