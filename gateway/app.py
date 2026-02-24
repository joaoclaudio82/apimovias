import asyncio
import copy
import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import JSONResponse, Response

AI_API_BASE_URL = os.getenv("AI_API_BASE_URL", "http://ai:8000").rstrip("/")
EXTRACTOR_API_BASE_URL = os.getenv(
    "EXTRACTOR_API_BASE_URL",
    "http://extractor:8000",
).rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("GATEWAY_TIMEOUT_SECONDS", "30"))

ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


app = FastAPI(
    title="Movias Gateway API",
    version="1.0.0",
    description="Gateway para AI API e Extractor API com documentação unificada.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _build_upstream_url(base_url: str, path: str, query: str) -> str:
    clean_path = path.lstrip("/")
    if clean_path:
        base = f"{base_url}/{clean_path}"
    else:
        base = base_url
    if query:
        return f"{base}?{query}"
    return base


async def _proxy_request(request: Request, base_url: str, path: str) -> Response:
    upstream_url = _build_upstream_url(base_url, path, request.url.query)
    headers = dict(request.headers)
    headers.pop("host", None)

    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            upstream_response = await client.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                content=body,
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erro ao conectar no upstream {base_url}: {exc}",
        ) from exc

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


def _rewrite_component_refs(node: Any, ref_mapping: Dict[str, str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str) and value in ref_mapping:
                node[key] = ref_mapping[value]
            else:
                _rewrite_component_refs(value, ref_mapping)
    elif isinstance(node, list):
        for item in node:
            _rewrite_component_refs(item, ref_mapping)


def _namespace_openapi_spec(spec: Dict[str, Any], namespace: str) -> Dict[str, Any]:
    namespaced = copy.deepcopy(spec)
    ref_mapping: Dict[str, str] = {}

    components = namespaced.get("components", {})
    if isinstance(components, dict):
        for section_name, section_items in components.items():
            if not isinstance(section_items, dict):
                continue

            renamed_items: Dict[str, Any] = {}
            for item_name, item_value in section_items.items():
                namespaced_name = f"{namespace}_{item_name}"
                old_ref = f"#/components/{section_name}/{item_name}"
                new_ref = f"#/components/{section_name}/{namespaced_name}"
                ref_mapping[old_ref] = new_ref
                renamed_items[namespaced_name] = item_value

            components[section_name] = renamed_items

    _rewrite_component_refs(namespaced, ref_mapping)

    paths = namespaced.get("paths", {})
    if isinstance(paths, dict):
        for path_item in paths.values():
            if not isinstance(path_item, dict):
                continue
            for method_item in path_item.values():
                if not isinstance(method_item, dict):
                    continue
                operation_id = method_item.get("operationId")
                if operation_id:
                    method_item["operationId"] = f"{namespace}_{operation_id}"

    return namespaced


def _prefixed_paths(spec: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    prefixed: Dict[str, Any] = {}
    for path, definition in spec.get("paths", {}).items():
        normalized_path = path if path.startswith("/") else f"/{path}"
        prefixed[f"{prefix}{normalized_path}"] = definition
    return prefixed


def _merge_tags(*specs: Dict[str, Any]) -> list[Dict[str, Any]]:
    merged_tags: list[Dict[str, Any]] = []
    seen: set[str] = set()

    for spec in specs:
        for tag in spec.get("tags", []):
            if not isinstance(tag, dict):
                continue
            name = tag.get("name")
            if not name or name in seen:
                continue
            merged_tags.append(tag)
            seen.add(name)

    return merged_tags


async def _fetch_openapi(client: httpx.AsyncClient, base_url: str) -> Dict[str, Any]:
    response = await client.get(f"{base_url}/openapi.json")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail=f"OpenAPI inválido recebido de {base_url}",
        )
    return payload


async def _build_merged_openapi() -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            ai_raw, extractor_raw = await asyncio.gather(
                _fetch_openapi(client, AI_API_BASE_URL),
                _fetch_openapi(client, EXTRACTOR_API_BASE_URL),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Falha ao obter OpenAPI dos serviços internos: {exc}",
        ) from exc

    ai_spec = _namespace_openapi_spec(ai_raw, "ai")
    extractor_spec = _namespace_openapi_spec(extractor_raw, "extractor")

    merged: Dict[str, Any] = {
        "openapi": ai_spec.get("openapi", "3.1.0"),
        "info": {
            "title": "Movias Gateway API",
            "version": "1.0.0",
        },
        "servers": [{"url": "/"}],
        "paths": {},
        "components": {},
        "tags": _merge_tags(ai_spec, extractor_spec),
    }

    merged["paths"].update(_prefixed_paths(ai_spec, "/ai"))
    merged["paths"].update(_prefixed_paths(extractor_spec, "/extractor"))

    for spec in (ai_spec, extractor_spec):
        components = spec.get("components", {})
        if not isinstance(components, dict):
            continue
        for section_name, section_items in components.items():
            if not isinstance(section_items, dict):
                continue
            merged["components"].setdefault(section_name, {})
            merged["components"][section_name].update(section_items)

    return merged


@app.get("/", tags=["health"])
async def root() -> Dict[str, str]:
    return {'status': 'ok'}


@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        checks = await asyncio.gather(
            client.get(f"{AI_API_BASE_URL}/health"),
            client.get(f"{EXTRACTOR_API_BASE_URL}/health"),
            return_exceptions=True,
        )

    ai_ok = isinstance(checks[0], httpx.Response) and checks[0].status_code < 500
    extractor_ok = (
        isinstance(checks[1], httpx.Response) and checks[1].status_code < 500
    )
    overall_ok = ai_ok and extractor_ok

    payload = {
        "status": "ok" if overall_ok else "degraded",
        "upstreams": {
            "ai_api": {
                "base_url": AI_API_BASE_URL,
                "ok": ai_ok,
                "status_code": checks[0].status_code if ai_ok else None,
            },
            "extractor_api": {
                "base_url": EXTRACTOR_API_BASE_URL,
                "ok": extractor_ok,
                "status_code": checks[1].status_code if extractor_ok else None,
            },
        },
    }
    return JSONResponse(status_code=200 if overall_ok else 503, content=payload)


@app.get("/openapi.json", include_in_schema=False)
async def openapi() -> JSONResponse:
    return JSONResponse(await _build_merged_openapi())


@app.get("/docs", include_in_schema=False)
async def docs() -> Response:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Movias Gateway Docs",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc() -> Response:
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Movias Gateway ReDoc",
    )


@app.api_route("/ai", methods=ALL_METHODS, include_in_schema=False)
@app.api_route("/ai/{path:path}", methods=ALL_METHODS, include_in_schema=False)
async def proxy_ai(request: Request, path: str = "") -> Response:
    return await _proxy_request(request, AI_API_BASE_URL, path)


@app.api_route("/extractor", methods=ALL_METHODS, include_in_schema=False)
@app.api_route(
    "/extractor/{path:path}",
    methods=ALL_METHODS,
    include_in_schema=False,
)
async def proxy_extractor(request: Request, path: str = "") -> Response:
    return await _proxy_request(request, EXTRACTOR_API_BASE_URL, path)
