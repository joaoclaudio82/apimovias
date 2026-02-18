.PHONY: api

api:
	uvicorn api.main:app --reload
