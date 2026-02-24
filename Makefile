PYTHON ?= python3
AI_PORT ?= 8010
EXTRACTOR_PORT ?= 8000
SHARED_CSV_PATH ?= $(abspath extractor/data/movias.csv)
AI_DATABASE_URL_LOCAL ?= sqlite+aiosqlite:///database.db

ENV_LOAD = set -a; [ -f .env ] && . ./.env; set +a;

.PHONY: ai extractor up

ai:
	@$(ENV_LOAD) \
	cd ai && \
	PYTHONPATH=. \
	DATABASE_URL="$${AI_DATABASE_URL_LOCAL:-$(AI_DATABASE_URL_LOCAL)}" \
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port $${AI_API_PORT:-$(AI_PORT)}

extractor:
	@$(ENV_LOAD) \
	cd extractor && \
	PYTHONPATH=. \
	AI_INTEGRATION_ENABLED="$${AI_INTEGRATION_ENABLED:-true}" \
	AI_API_BASE_URL="$${AI_API_BASE_URL:-http://127.0.0.1:$${AI_API_PORT:-$(AI_PORT)}}" \
	AI_SHARED_CSV_PATH="$(SHARED_CSV_PATH)" \
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port $${EXTRACTOR_API_PORT:-$(EXTRACTOR_PORT)}

up:
	$(MAKE) -j2 ai extractor
