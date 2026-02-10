.PHONY: api

api:
	uvicorn api:app --reload
