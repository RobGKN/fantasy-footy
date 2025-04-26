# Makefile for AFL Fantasy Breakout Project

PYTHON=python
PYTHONPATH=.

preprocess:
	@echo "Running data preprocessing..."
	PYTHONPATH=. python src/data_preprocessing.py
	
run-training:
	@echo "Running training pipeline..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/run_training.py

test:
	@echo "Running tests..."
	PYTHONPATH=$(PYTHONPATH) pytest tests/

lint:
	@echo "Checking code style with flake8..."
	flake8 src/ tests/

format:
	@echo "Auto-formatting with black..."
	black src/ tests/