PYTHON ?= python

.PHONY: install-dev test build clean

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

build:
	rm -rf build dist
	$(PYTHON) -m build

clean:
	rm -rf build dist .pytest_cache
