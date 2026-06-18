"""Lista de veículos com filtros."""

import pandas as pd
import streamlit as st

import api_client as api
from helpers import QUALITY_LABELS, quality_label

st.title("🚗 Veículos")

# ── Filters ───────────────────────────────────────────────────

col_f1, col_f2, col_f3 = st.columns(3)

quality_options = {v: k for k, v in QUALITY_LABELS.items()}
quality_options_display = ["Todos"] + list(quality_options.keys())

with col_f1:
    sel_q = st.selectbox("Qualidade", quality_options_display, index=0)
with col_f2:
    target_options = ["Todos", "KM + H", "KM only", "H only"]
    sel_target = st.selectbox("Targets", target_options, index=0)
with col_f3:
    search_id = st.text_input("Buscar ID do Veículo", "")

hide_empty = st.checkbox("Ocultar veículos com série vazia", value=True)

# Identificar single-target
def _available_targets_label(row):
    has_km = row.get("dt_inicio_km") is not None
    has_h = row.get("dt_inicio_h") is not None
    if has_km and has_h:
        return "KM + H"
    if has_km:
        return "KM only"
    if has_h:
        return "H only"
    return "—"

q_val = quality_options.get(sel_q)

# ── Load data ─────────────────────────────────────────────────

try:
    vehicles = api.list_vehicles(quality=q_val)
except Exception as e:
    st.error(f"Erro ao carregar veículos: {e}")
    st.stop()

if not vehicles:
    st.info("Nenhum veículo encontrado com os filtros aplicados.")
    st.stop()

df = pd.DataFrame(vehicles)

# Filtrar veículos com qualidade vazia
if hide_empty:
    import math
    df = df[df["quality"].apply(lambda x: x is not None and not (isinstance(x, float) and math.isnan(x)) and x != 3)]

# Apply search filter
if search_id.strip():
    try:
        ids = [int(x.strip()) for x in search_id.split(",")]
        df = df[df["veiculo_id"].isin(ids)]
    except ValueError:
        st.warning("IDs devem ser numéricos (ex: 4, 11, 38)")

if df.empty:
    st.info("Nenhum veículo encontrado.")
    st.stop()

# ── Format table ──────────────────────────────────────────────

df["quality_label"] = df["quality"].apply(quality_label)
df["targets"] = df.apply(_available_targets_label, axis=1)

# Apply target filter
if sel_target != "Todos":
    df = df[df["targets"] == sel_target]

if df.empty:
    st.info("Nenhum veículo encontrado.")
    st.stop()

display_cols = [
    "veiculo_id",
    "quality_label",
    "targets",
    "dt_inicio_km",
    "dt_fim_km",
    "upper_km",
    "dt_inicio_h",
    "dt_fim_h",
    "upper_h",
]
rename = {
    "veiculo_id": "ID do Veículo",
    "quality_label": "Qualidade",
    "targets": "Targets",
    "dt_inicio_km": "Início KM",
    "dt_fim_km": "Fim KM",
    "upper_km": "Upper KM",
    "dt_inicio_h": "Início H",
    "dt_fim_h": "Fim H",
    "upper_h": "Upper H",
}

st.metric("Total", len(df))
st.dataframe(
    df[display_cols].rename(columns=rename),
    use_container_width=True,
    hide_index=True,
    height=600,
)

# ── Quick nav ─────────────────────────────────────────────────

st.divider()
st.caption("Clique em um ID para ir ao detalhe:")
cols = st.columns(min(10, len(df)))
for i, vid in enumerate(df["veiculo_id"].head(20)):
    with cols[i % len(cols)]:
        if st.button(str(vid), key=f"nav_{vid}"):
            st.session_state["detail_veiculo_id"] = int(vid)
            st.switch_page("pages/vehicle_detail.py")
