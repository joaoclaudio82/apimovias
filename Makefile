PYTHON ?= python3
AI_PORT ?= 8010
EXTRACTOR_PORT ?= 8000
GATEWAY_PORT ?= 8080
AI_DATABASE_URL_LOCAL ?= sqlite+aiosqlite:///database.db

ENV_LOAD = set -a; [ -f .env ] && . ./.env; set +a

.PHONY: ai extractor gateway up

ai:
	@$(ENV_LOAD); \
	cd ai; \
	PYTHONPATH=. \
	DATABASE_URL="$${AI_DATABASE_URL_LOCAL:-$(AI_DATABASE_URL_LOCAL)}" \
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port "$${AI_API_PORT:-$(AI_PORT)}"

extractor:
	@$(ENV_LOAD); \
	cd extractor; \
	PYTHONPATH=. \
	LOG_LEVEL="$${EXTRACTOR_LOG_LEVEL:-INFO}" \
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port "$${EXTRACTOR_API_PORT:-$(EXTRACTOR_PORT)}"

gateway:
	@$(ENV_LOAD); \
	cd gateway; \
	PYTHONPATH=. \
	AI_API_BASE_URL="http://127.0.0.1:$${AI_API_PORT:-$(AI_PORT)}" \
	EXTRACTOR_API_BASE_URL="http://127.0.0.1:$${EXTRACTOR_API_PORT:-$(EXTRACTOR_PORT)}" \
	GATEWAY_TIMEOUT_SECONDS="$${GATEWAY_TIMEOUT_SECONDS:-}" \
	$(PYTHON) -m uvicorn app:app --host 127.0.0.1 --port "$${GATEWAY_PORT:-$(GATEWAY_PORT)}"

up:
	@$(MAKE) --no-print-directory -j3 ai extractor gateway
