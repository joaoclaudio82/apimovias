"""Movias App – ponto de entrada."""

import streamlit as st

st.set_page_config(
    page_title="Movias",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("pages/home.py", title="Início", icon="🏠", default=True)
vehicles = st.Page("pages/vehicles.py", title="Veículos", icon="🚗")
vehicle_detail = st.Page("pages/vehicle_detail.py", title="Detalhe do Veículo", icon="🔍")
predictions = st.Page("pages/predictions.py", title="Predições", icon="📈")
ingestion = st.Page("pages/ingestion.py", title="Ingestão", icon="📤")
pipeline = st.Page("pages/pipeline.py", title="Pipeline / Tasks", icon="⚙️")
training_pipeline = st.Page("pages/training_pipeline.py", title="Pipeline de Treinamento", icon="🏋️")
models = st.Page("pages/models.py", title="Modelos", icon="🧠")
logs = st.Page("pages/logs.py", title="Logs", icon="📋")

pg = st.navigation(
    {
        "Geral": [home],
        "Veículos": [vehicles, vehicle_detail, predictions],
        "Operações": [ingestion, pipeline, training_pipeline, models],
        "Monitoramento": [logs],
    }
)
pg.run()
