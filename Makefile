.PHONY: install test lint demo train drift serve docker

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

demo:
	python -m parkwaits.cli init-demo --rows 5000

train:
	python -m parkwaits.cli train

drift:
	python -m parkwaits.cli drift

serve:
	python -m parkwaits.cli serve

docker:
	docker build -t parkwaits .
