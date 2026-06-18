"""Cliente HTTP para a API do Movias.

Suporta acesso direto (API_URL=http://localhost:8010, API_PREFIX="")
e acesso via gateway (API_URL=http://localhost:8080, API_PREFIX="/ai").
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = os.getenv("API_URL", "http://localhost:8010")
API_PREFIX = os.getenv("API_PREFIX", "")  # "/ai" when behind gateway
TIMEOUT = float(os.getenv("API_TIMEOUT", "120"))

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)
    return _client


def _p(path: str) -> str:
    """Prepend API_PREFIX to path."""
    return f"{API_PREFIX}{path}"


# ── helpers ───────────────────────────────────────────────────


def _get(path: str, **params) -> Any:
    resp = _get_client().get(path, params=params)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, json: Any = None, **kwargs) -> Any:
    resp = _get_client().post(path, json=json, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _put(path: str, json: Any = None) -> Any:
    resp = _get_client().put(path, json=json)
    resp.raise_for_status()
    return resp.json()


# ── Profiling ─────────────────────────────────────────────────


def list_vehicles(quality: int | None = None) -> List[Dict]:
    params: Dict[str, Any] = {}
    if quality is not None:
        params["quality"] = quality
    return _get(_p("/profiling/vehicles"), **params)


def get_vehicle_info(veiculo_id: int, target: str) -> Dict:
    return _get(_p(f"/profiling/vehicle/{veiculo_id}"), target=target)


def get_vehicle_history(veiculo_id: int, target: str) -> List[Dict] | None:
    """Retorna a série histórica diária de um veículo para o target."""
    try:
        return _get(_p(f"/profiling/vehicle/{veiculo_id}/history"), target=target)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise


def get_profile_metadata(last: bool = False) -> List[Dict]:
    return _get(_p("/profiling/metadata"), last=str(last).lower())


def ingest_csv(file_bytes: bytes, filename: str) -> Dict:
    return _post(
        _p("/profiling/ingest"),
        files={"file": (filename, file_bytes, "text/csv")},
    )


# ── Prediction ────────────────────────────────────────────────


def predict(target: str, vehicle_ids: List[int] | None = None) -> Dict:
    body: Dict[str, Any] = {"target": target}
    if vehicle_ids is not None:
        body["vehicle_ids"] = vehicle_ids
    return _post(_p("/prediction/predict"), json=body)


def backtest(veiculo_id: int, target: str, date_from: str | None = None, date_to: str | None = None, max_heads: int | None = None) -> Dict | None:
    try:
        params: Dict[str, Any] = {"target": target}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if max_heads is not None:
            params["max_heads"] = max_heads
        return _get(_p(f"/prediction/backtest/{veiculo_id}"), **params)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise


# ── Pipeline ──────────────────────────────────────────────────


def list_pipeline_runs(
    step: str | None = None,
    target: str | None = None,
    status: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> List[Dict]:
    params: Dict[str, Any] = {"offset": offset, "limit": limit}
    if step:
        params["step"] = step
    if target:
        params["target"] = target
    if status:
        params["status"] = status
    return _get(_p("/pipeline/runs"), **params)


def get_pipeline_run(run_id: int) -> Dict:
    return _get(_p(f"/pipeline/runs/{run_id}"))


def list_active_models() -> List[Dict]:
    return _get(_p("/pipeline/models/active"))


def list_available_models(target: str) -> List[Dict]:
    return _get(_p("/pipeline/models/available"), target=target)


def set_active_model(target: str, version_id: str) -> Dict:
    return _put(_p("/pipeline/models/active"), json={
        "target": target,
        "version_id": version_id,
    })


# ── Pipeline de Treinamento ───────────────────────────────────


def start_segmentation(target: str, data_path: str | None = None) -> Dict:
    body: Dict[str, Any] = {"target": target}
    if data_path:
        body["data_path"] = data_path
    return _post(_p("/pipeline/segmentation"), json=body)


def start_profiles(target: str) -> Dict:
    return _post(_p("/pipeline/profiles"), json={"target": target})


def start_datasets(target: str, use_cache: bool = True) -> Dict:
    return _post(_p("/pipeline/datasets"), json={"target": target, "use_cache": use_cache})


def start_optimization(target: str, model_type: str = "moe") -> Dict:
    return _post(_p("/pipeline/optimization"), json={"target": target, "model_type": model_type})


def start_training(target: str, model_type: str = "moe") -> Dict:
    return _post(_p("/pipeline/training"), json={"target": target, "model_type": model_type})


def start_full_pipeline(
    target: str,
    data_path: str | None = None,
    model_type: str = "moe",
    final_step: str = "optimization",
) -> List[Dict]:
    body: Dict[str, Any] = {"target": target, "model_type": model_type, "final_step": final_step}
    if data_path:
        body["data_path"] = data_path
    return _post(_p("/pipeline/full"), json=body)


# ── Health ─────────────────────────────────────────────────────


def health() -> Dict:
    return _get(_p("/health"))
