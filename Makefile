PYTHON ?= python3
AI_PORT ?= 8010
EXTRACTOR_PORT ?= 8000
SHARED_CSV_PATH ?= $(abspath extractor/data/movias.csv)

.PHONY: ai extractor up

ai:
	cd ai && PYTHONPATH=. $(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port $(AI_PORT)

extractor:
	cd extractor && \
		PYTHONPATH=. \
		AI_INTEGRATION_ENABLED=true \
		AI_API_BASE_URL=http://127.0.0.1:$(AI_PORT) \
		AI_SHARED_CSV_PATH=$(SHARED_CSV_PATH) \
		$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port $(EXTRACTOR_PORT)

up:
	$(MAKE) -j2 ai extractor
