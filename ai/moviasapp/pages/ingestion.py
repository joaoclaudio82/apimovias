"""Ingestão – upload de CSV e execução de ingestão + predição."""

import streamlit as st

import api_client as api

st.title("📤 Ingestão de Dados")

st.markdown("""
Faça upload de um CSV com as colunas obrigatórias:
- `veiculo_id` (int)
- `data` (YYYY-MM-DD)
- `h_dia_clean` (float)
- `km_dia_clean` (float)

O sistema irá:
1. Inserir os dados novos em `daily_activity`
2. Executar o pipeline de profiling
3. Executar predição para todos os veículos (H e KM)

Tudo numa única transação atómica.
""")

st.divider()

uploaded = st.file_uploader("Selecionar CSV", type=["csv"])

if uploaded is not None:
    st.caption(f"Arquivo: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

    # Preview
    import pandas as pd
    try:
        df_preview = pd.read_csv(uploaded)
        uploaded.seek(0)  # reset for submission

        st.subheader("Pré-visualização")
        st.dataframe(df_preview.head(20), use_container_width=True, hide_index=True)

        required = {"veiculo_id", "data", "h_dia_clean", "km_dia_clean"}
        missing = required - set(df_preview.columns)
        if missing:
            st.error(f"Colunas obrigatórias faltando: {sorted(missing)}")
            st.stop()

        st.success(f"✅ {len(df_preview)} linhas, {df_preview['veiculo_id'].nunique()} veículos")

    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    st.divider()

    if st.button("🚀 Submeter ingestão", type="primary"):
        with st.spinner("Enviando para o servidor..."):
            try:
                result = api.ingest_csv(uploaded.read(), uploaded.name)
                st.success(
                    f"Ingestão submetida com sucesso! "
                    f"**run_id = {result.get('run_id')}**"
                )
                st.info(result.get("message", ""))
                st.caption("Acompanhe o progresso na página **Pipeline / Tasks**.")
            except Exception as e:
                st.error(f"Falha na submissão: {e}")

# ── Profile metadata ─────────────────────────────────────────

st.divider()
st.subheader("Perfil atual")

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
        st.info("Nenhum perfil carregado.")
except Exception as e:
    st.warning(f"Erro: {e}")
