"""Predições – visualizar predições persistidas."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api_client as api
from helpers import quality_label

st.title("📈 Predições")

# ── Inputs ────────────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    target = st.selectbox("Target", ["km", "h"])
with col2:
    vehicle_ids_str = st.text_input(
        "IDs de veículos (vazio = todos)",
        placeholder="4, 11, 38",
    )

vehicle_ids = None
if vehicle_ids_str.strip():
    try:
        vehicle_ids = [int(x.strip()) for x in vehicle_ids_str.split(",")]
    except ValueError:
        st.error("IDs devem ser numéricos.")
        st.stop()

if st.button("Consultar", type="primary"):
    st.session_state["pred_loaded"] = True
    st.session_state["pred_target"] = target
    st.session_state["pred_vehicle_ids"] = vehicle_ids

if not st.session_state.get("pred_loaded"):
    st.info("Selecione o target e clique em **Consultar**.")
    st.stop()

# Usar valores do session_state
target = st.session_state.get("pred_target", target)
vehicle_ids = st.session_state.get("pred_vehicle_ids", vehicle_ids)

# ── Load ──────────────────────────────────────────────────────

with st.spinner("Carregando predições..."):
    try:
        data = api.predict(target, vehicle_ids)
    except Exception as e:
        st.error(f"Erro: {e}")
        st.stop()

# ── Quality ───────────────────────────────────────────────────

vq = data.get("vehicle_quality", [])
if vq:
    with st.expander("Qualidade dos veículos", expanded=False):
        df_q = pd.DataFrame(vq)
        st.dataframe(df_q, use_container_width=True, hide_index=True)

# ── Daily ─────────────────────────────────────────────────────

daily = data.get("predictions_daily", [])
if daily:
    st.subheader(f"Predições Diárias ({len(daily)} registros)")
    df_daily = pd.DataFrame(daily)
    df_daily["data"] = pd.to_datetime(df_daily["data"])

    # Chart: one line per vehicle
    vids = df_daily["veiculo_id"].unique()
    if len(vids) <= 20:
        fig = go.Figure()
        for vid in sorted(vids):
            sub = df_daily[df_daily["veiculo_id"] == vid].sort_values("data")
            fig.add_trace(go.Scatter(
                x=sub["data"], y=sub["prediction"],
                name=str(vid), mode="lines+markers",
            ))
        fig.update_layout(
            height=400, margin=dict(t=30, b=30),
            xaxis_title="Data", yaxis_title=target.upper(),
            legend_title="Veículo",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(f"{len(vids)} veículos — gráfico omitido (use filtro de IDs)")

    with st.expander("Tabela completa"):
        st.dataframe(df_daily, use_container_width=True, hide_index=True)
else:
    st.info("Sem predições diárias.")

# ── Heads ─────────────────────────────────────────────────────

heads = data.get("predictions_heads", [])
if heads:
    st.subheader(f"Predições Heads ({len(heads)} registros)")
    df_heads = pd.DataFrame(heads)
    with st.expander("Tabela completa"):
        st.dataframe(df_heads, use_container_width=True, hide_index=True)
else:
    st.info("Sem predições heads.")

# ── Type probabilities ────────────────────────────────────────

tp = data.get("type_probabilities", [])
if tp:
    st.subheader("Probabilidades de tipo")
    # Filtrar apenas features que começam com type_ (probabilidades)
    for item in tp:
        probs = {k: v for k, v in item.get("probabilities", {}).items() if k.startswith("type_")}
        if probs:
            vid = item["veiculo_id"]
            st.markdown(f"**Veículo {vid}**")
            prob_df = pd.DataFrame(
                [{"Tipo": k.replace("type_", ""), "Probabilidade": v} for k, v in sorted(probs.items())]
            )
            fig = go.Figure(go.Bar(
                x=prob_df["Tipo"],
                y=prob_df["Probabilidade"],
                marker_color="#636EFA",
                text=prob_df["Probabilidade"].round(3),
                textposition="outside",
            ))
            fig.update_layout(
                height=250,
                margin=dict(t=20, b=20),
                yaxis_title="Probabilidade",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Features de classificação de tipo ─────────────────────────

if tp:
    has_class_features = False
    for item in tp:
        class_feats = {k: v for k, v in item.get("probabilities", {}).items() if not k.startswith("type_")}
        if class_feats:
            if not has_class_features:
                st.subheader("Features para classificação de tipo (km ou h)")
                has_class_features = True
            vid = item["veiculo_id"]
            feat_df = pd.DataFrame(
                [{"Feature": k, "Valor": round(v, 4) if isinstance(v, float) else v} for k, v in sorted(class_feats.items())]
            )
            st.markdown(f"**Veículo {vid}**")
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ── Not found ─────────────────────────────────────────────────

nf = data.get("not_found", [])
if nf:
    st.warning(f"{len(nf)} veículos não encontrados")
    st.dataframe(pd.DataFrame(nf), use_container_width=True, hide_index=True)
