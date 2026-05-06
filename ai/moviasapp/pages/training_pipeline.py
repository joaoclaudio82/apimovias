"""Pipeline de Treinamento – iniciar etapas e acompanhar execuções."""

from pathlib import Path

import pandas as pd
import streamlit as st

import api_client as api

st.title("🏋️ Pipeline de Treinamento")

# ── Diretório para uploads ────────────────────────────────────

_DATASETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "datasets"
_UPLOAD_DIR = _DATASETS_DIR / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(ttl=30)
def _list_csv_files() -> list[str]:
    """Lista todos os CSVs recursivamente na pasta datasets/."""
    if not _DATASETS_DIR.exists():
        return []
    return sorted(str(p) for p in _DATASETS_DIR.rglob("*.csv"))


st.divider()

# ── Pipeline completo ─────────────────────────────────────────

st.subheader("Pipeline completo")
st.caption("Executa todas as etapas em sequência: Segmentação → Perfis → Dataset → Otimização")

col_target, col_model, col_step = st.columns(3)
with col_target:
    full_target = st.selectbox("Target", ["km", "h"], key="full_target")
with col_model:
    model_type = st.selectbox("Tipo de modelo", ["moe", "multihead"], key="full_model_type")
with col_step:
    final_step = st.selectbox("Etapa final", ["optimization", "training"], key="full_final_step",
                              format_func=lambda s: "Otimização" if s == "optimization" else "Treinamento")

csv_files = _list_csv_files()
data_path = st.selectbox(
    "Arquivo de dados (segmentação)",
    csv_files,
    index=0 if csv_files else None,
    format_func=lambda p: Path(p).relative_to(_DATASETS_DIR).as_posix(),
    key="full_data_file",
)

with st.expander("📤 Enviar novo arquivo para a pasta 'datasets'"):
    uploaded = st.file_uploader("Selecionar CSV", type=["csv"], key="training_csv")
    if uploaded is not None:
        save_path = _UPLOAD_DIR / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
        st.success(f"Arquivo salvo em `datasets/uploads/{uploaded.name}`")
        st.caption("Atualize a página para vê-lo no seletor acima.")

if st.button("▶️ Iniciar pipeline completo", type="primary"):
    try:
        with st.spinner("Iniciando pipeline completo..."):
            runs = api.start_full_pipeline(
                target=full_target,
                data_path=data_path,
                model_type=model_type,
                final_step=final_step,
            )
        st.success(f"Pipeline iniciado — {len(runs)} etapas enfileiradas.")
        for r in runs:
            st.write(f"• **{r['step']}** (ID: {r['id']}) — {r['status']}")
    except Exception as e:
        st.error(f"Erro: {e}")

st.divider()

# ── Etapas individuais ────────────────────────────────────────

st.subheader("Etapas individuais")

tabs = st.tabs(["Segmentação", "Perfis", "Dataset", "Otimização", "Treinamento"])

with tabs[0]:
    st.caption("Segmentação de veículos por comportamento")
    seg_data_path = st.selectbox(
        "Arquivo de dados",
        csv_files,
        index=0 if csv_files else None,
        format_func=lambda p: Path(p).relative_to(_DATASETS_DIR).as_posix(),
        key="seg_data_file",
    )
    if st.button("Iniciar Segmentação", key="btn_seg"):
        try:
            with st.spinner("Iniciando segmentação..."):
                r = api.start_segmentation("km", data_path=seg_data_path)
            st.success(f"Segmentação iniciada (ID: {r['id']})")
        except Exception as e:
            st.error(f"Erro: {e}")

with tabs[1]:
    st.caption("Geração de perfis dos veículos")
    profiles_target = st.selectbox("Target", ["km", "h"], key="profiles_target")
    if st.button("Iniciar Perfis", key="btn_profiles"):
        try:
            with st.spinner("Iniciando geração de perfis..."):
                r = api.start_profiles(profiles_target)
            st.success(f"Perfis iniciados (ID: {r['id']})")
        except Exception as e:
            st.error(f"Erro: {e}")

with tabs[2]:
    st.caption("Geração de datasets de treinamento")
    dataset_target = st.selectbox("Target", ["km", "h"], key="dataset_target")
    dataset_use_cache = st.checkbox("Utilizar cache existente", value=True, key="dataset_use_cache")
    if st.button("Iniciar Dataset", key="btn_dataset"):
        try:
            with st.spinner("Iniciando geração de dataset..."):
                r = api.start_datasets(dataset_target, use_cache=dataset_use_cache)
            st.success(f"Dataset iniciado (ID: {r['id']})")
        except Exception as e:
            st.error(f"Erro: {e}")

with tabs[3]:
    st.caption("Otimização de hiperparâmetros e treinamento do modelo")
    opt_target = st.selectbox("Target", ["km", "h"], key="opt_target")
    opt_model_type = st.selectbox("Tipo de modelo", ["moe", "multihead"], key="opt_model_type")
    if st.button("Iniciar Otimização", key="btn_opt"):
        try:
            with st.spinner("Iniciando otimização..."):
                r = api.start_optimization(opt_target, model_type=opt_model_type)
            st.success(f"Otimização iniciada (ID: {r['id']})")
        except Exception as e:
            st.error(f"Erro: {e}")

with tabs[4]:
    st.caption("Treinamento direto com hiperparâmetros da configuração (model_config.yaml / training_config.yaml)")
    train_target = st.selectbox("Target", ["km", "h"], key="train_target")
    train_model_type = st.selectbox("Tipo de modelo", ["moe", "multihead"], key="train_model_type")
    if st.button("Iniciar Treinamento", key="btn_train"):
        try:
            with st.spinner("Iniciando treinamento..."):
                r = api.start_training(train_target, model_type=train_model_type)
            st.success(f"Treinamento iniciado (ID: {r['id']})")
        except Exception as e:
            st.error(f"Erro: {e}")

st.divider()

# ── Últimas execuções ─────────────────────────────────────────

col_header, col_btn = st.columns([3, 1])
with col_header:
    st.subheader("Últimas execuções")
with col_btn:
    st.write("")  # espaçamento vertical
    if st.button("🔄 Atualizar", key="btn_refresh_runs"):
        st.session_state["_runs_refresh"] = True

_TRAINING_STEPS = {"segmentation", "profiles", "dataset", "optimization", "training"}

try:
    runs = api.list_pipeline_runs(limit=50)
    if runs:
        df = pd.DataFrame(runs)
        # Filtrar apenas steps de treinamento
        if "step" in df.columns:
            df = df[df["step"].isin(_TRAINING_STEPS)]
        df = df.head(10)
        if not df.empty:
            status_emoji = {"pending": "🟡", "running": "🔵", "completed": "🟢", "failed": "🔴"}
            if "status" in df.columns:
                df["status"] = df["status"].apply(lambda s: f"{status_emoji.get(s, '⚪')} {s}")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhuma execução de treinamento registrada.")
    else:
        st.info("Nenhuma execução registrada.")
except Exception as e:
    st.error(f"Erro ao carregar execuções: {e}")
