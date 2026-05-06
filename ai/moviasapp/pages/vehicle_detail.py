"""Detalhe do veículo – perfil, metadados, classificação, backtest."""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api_client as api
from helpers import quality_label

st.title("🔍 Detalhe do Veículo")

# ── Inputs ────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1])
with col1:
    veiculo_id = st.number_input(
        "ID do Veículo",
        min_value=1,
        step=1,
        value=st.session_state.get("detail_veiculo_id", 1),
    )
with col2:
    target = st.selectbox("Target", ["km", "h"])

if st.button("Carregar", type="primary"):
    st.session_state["detail_loaded"] = True
    st.session_state["detail_veiculo_id"] = int(veiculo_id)
    st.session_state["detail_target"] = target

if not st.session_state.get("detail_loaded"):
    st.info("Insira o ID e clique em **Carregar**.")
    st.stop()

# Usar valores do session_state
veiculo_id = st.session_state.get("detail_veiculo_id", veiculo_id)
target = st.session_state.get("detail_target", target)

# ── Load data ─────────────────────────────────────────────────

try:
    info = api.get_vehicle_info(veiculo_id, target)
except Exception as e:
    st.error(f"Erro: {e}")
    st.stop()

# ── Metadata ──────────────────────────────────────────────────

st.subheader("Metadados")

meta = info.get("metadata")
if meta:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Início", meta["dt_inicio"])
    c2.metric("Fim", meta["dt_fim"])
    c3.metric("Upper", f"{meta['upper']:.2f}" if meta.get("upper") else "—")
    c4.metric("Qualidade", quality_label(meta.get("quality")))

    if meta.get("quality_reason"):
        st.caption(f"Motivo: {meta['quality_reason']}")
else:
    st.warning("Sem metadados para este veículo/target.")

# ── Profile features ─────────────────────────────────────────

st.divider()
st.subheader("Perfil de Features")

profile = info.get("profile")
if profile:
    # Classificar features por categoria
    suffix = f"_{target}"
    type_prob_features = {k: v for k, v in profile.items() if k.startswith("type_")}
    cluster_features = {k: v for k, v in profile.items() if k.startswith("cluster_")}
    day_features = {k: v for k, v in profile.items() if k.startswith("day_")}
    seg_features = {k: v for k, v in profile.items() if k.startswith("seg_")}
    phase_features = {k: v for k, v in profile.items() if k.startswith("phase_")}
    cycle_features = {k: v for k, v in profile.items() if k.startswith("cycle_")}

    def _strip(name: str) -> str:
        """Remove prefixo (seg_, phase_, cycle_) e sufixo (_km, _h)."""
        # Remover sufixo do target
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        # Remover prefixo
        for prefix in ("seg_", "phase_", "cycle_", "day_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name

    # ── Classificação de tipo (probabilidades) ─────────────────

    if type_prob_features:
        st.markdown("**Classificação de tipo**")
        type_prob_df = pd.DataFrame(
            [{"Tipo": k.replace("type_", ""), "Probabilidade": v} for k, v in sorted(type_prob_features.items())],
        )

        fig_type = go.Figure(go.Bar(
            x=type_prob_df["Tipo"],
            y=type_prob_df["Probabilidade"],
            marker_color="#636EFA",
            text=type_prob_df["Probabilidade"].round(3),
            textposition="outside",
        ))
        fig_type.update_layout(
            height=300,
            margin=dict(t=30, b=30),
            yaxis_title="Probabilidade",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_type, use_container_width=True)

    # ── Segmentação (cluster) ─────────────────────────────────

    if cluster_features:
        st.markdown(f"**Segmentação (cluster {target.upper()})**")
        cluster_df = pd.DataFrame(
            [{"Cluster": k.replace(suffix, ""), "Probabilidade": v} for k, v in sorted(cluster_features.items())],
        )

        fig_cluster = go.Figure(go.Bar(
            x=cluster_df["Cluster"],
            y=cluster_df["Probabilidade"],
            marker_color="#EF553B",
            text=cluster_df["Probabilidade"].round(3),
            textposition="outside",
        ))
        fig_cluster.update_layout(
            height=300,
            margin=dict(t=30, b=30),
            yaxis_title="Probabilidade",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    # ── Features métricas (seg, phase, cycle) ─────────────────

    CATEGORY_LABELS = {
        "seg": "Atividade Operacional",
        "phase": "Fase Mensal",
        "cycle": "Ciclo Operacional",
    }

    metric_groups = [
        ("seg", seg_features),
        ("phase", phase_features),
        ("cycle", cycle_features),
    ]
    combined_rows = []
    for cat, feats in metric_groups:
        for k, v in sorted(feats.items()):
            combined_rows.append({
                "Tipo": CATEGORY_LABELS[cat],
                "Feature": _strip(k),
                "Valor": round(v, 4) if isinstance(v, float) else v,
            })

    if combined_rows:
        st.markdown(f"**Features ({target.upper()})**")
        feat_df = pd.DataFrame(combined_rows)
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=400)

    # ── Features por dia da semana (matrix) ───────────────────

    DAY_NAMES = {
        "1": "Segunda",
        "2": "Terça",
        "3": "Quarta",
        "4": "Quinta",
        "5": "Sexta",
        "6": "Sábado",
        "7": "Domingo",
    }

    if day_features:
        st.markdown(f"**Features por dia da semana ({target.upper()})**")
        # Formato: day_{1-7}_{stat}_{target} → extrair dia e stat
        import re
        day_data: dict[str, dict[str, float]] = {}
        for k, v in day_features.items():
            # Remove suffix target
            name = k[: -len(suffix)] if k.endswith(suffix) else k
            # Extrair: day_{dia}_{stat}
            m = re.match(r"day_(\d+)_(.+)", name)
            if m:
                dia, stat = m.group(1), m.group(2)
                day_data.setdefault(stat, {})[dia] = round(v, 4) if isinstance(v, float) else v

        if day_data:
            # Construir DataFrame com dias como colunas
            rows = []
            for stat in sorted(day_data.keys()):
                row = {"Feature": stat}
                for d in ("1", "2", "3", "4", "5", "6", "7"):
                    row[DAY_NAMES[d]] = day_data[stat].get(d, None)
                rows.append(row)
            day_df = pd.DataFrame(rows)
            st.dataframe(day_df, use_container_width=True, hide_index=True)
else:
    st.info("Sem perfil carregado para este veículo.")

# ── Backtest ──────────────────────────────────────────────────

st.divider()
st.subheader("Backtest")

# Buscar backtest completo para descobrir intervalo de datas
bt_full = api.backtest(veiculo_id, target)

if bt_full is None:
    st.info("Sem dados de backtest (nenhuma predição com valor real preenchido).")
else:
    all_daily = bt_full.get("daily", [])
    all_heads = bt_full.get("heads", [])

    # ── Filtros de período (baseados nos dados reais) ─────────

    if all_daily:
        dates = [date.fromisoformat(d["data"]) for d in all_daily]
        data_min, data_max = min(dates), max(dates)
    else:
        data_min = data_max = date.today()

    default_from = max(data_min, data_max - timedelta(days=30))

    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        bt_date_from = st.date_input(
            "Período inicial",
            value=default_from,
            min_value=data_min,
            max_value=data_max,
            format="DD/MM/YYYY",
        )
    with col_bt2:
        bt_date_to = st.date_input(
            "Período final",
            value=data_max,
            min_value=data_min,
            max_value=data_max,
            format="DD/MM/YYYY",
        )
    with col_bt3:
        bt_max_heads = st.number_input(
            "Máx. blocos semanais",
            min_value=1,
            value=min(4, len(all_heads)) if all_heads else 4,
            step=1,
        )

    # Filtrar daily localmente (dados já carregados)
    daily = [
        d for d in all_daily
        if str(bt_date_from) <= d["data"] <= str(bt_date_to)
    ]

    # Filtrar heads: mais recentes
    heads = all_heads[-bt_max_heads:] if all_heads else []

    # ── Gráfico diário ────────────────────────────────────────

    if daily:
        st.markdown("**Diário**")
        df_d = pd.DataFrame(daily)
        df_d["data"] = pd.to_datetime(df_d["data"])
        df_d = df_d.sort_values("data")

        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=df_d["data"], y=df_d["actual"],
            name="Real", mode="lines+markers", line=dict(color="#636EFA"),
        ))
        fig_d.add_trace(go.Scatter(
            x=df_d["data"], y=df_d["predicted"],
            name="Predição", mode="lines+markers", line=dict(color="#EF553B", dash="dash"),
        ))
        fig_d.update_layout(
            height=350,
            margin=dict(t=30, b=30),
            legend=dict(orientation="h", y=1.12),
            xaxis_title="Data",
            yaxis_title=target.upper(),
        )
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.info("Sem dados diários no período selecionado.")

    # ── Gráfico de blocos semanais ────────────────────────────

    if heads:
        st.markdown("**Blocos semanais (Heads)**")
        df_h = pd.DataFrame(heads)
        df_h["periodo"] = df_h["dt_inicio"] + " → " + df_h["dt_fim"]

        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=df_h["periodo"], y=df_h["actual"],
            name="Real", marker_color="#636EFA",
        ))
        fig_h.add_trace(go.Bar(
            x=df_h["periodo"], y=df_h["predicted"],
            name="Predição", marker_color="#EF553B",
        ))
        fig_h.update_layout(
            barmode="group",
            height=350,
            margin=dict(t=30, b=30),
            legend=dict(orientation="h", y=1.12),
            yaxis_title=target.upper(),
        )
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Sem blocos semanais disponíveis.")

    if not all_daily and not all_heads:
        st.info("Backtest retornado mas sem dados diários ou heads.")
