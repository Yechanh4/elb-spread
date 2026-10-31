PY ?= python3.12
VENV = .venv
BIN = $(VENV)/bin

.PHONY: venv install test live run fetch docker

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(BIN)/pip install -r requirements-dev.txt

test:
	$(BIN)/python -m pytest -q

live:
	$(BIN)/python -m pytest -q -m live

run:
	$(BIN)/uvicorn app.main:app --port 8780

fetch:
	$(BIN)/python -m data.cli refresh-all

docker:
	docker build -t elb-spread . && docker run --rm -p 8780:8780 -v elb-data:/data elb-spread
