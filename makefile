# Makefile for AFL Fantasy Breakout Project

PYTHON=python
PYTHONPATH=.

preprocess:
	@echo "Running data preprocessing..."
	PYTHONPATH=. python src/data_preprocessing.py
	
run-training:
	@echo "Running training pipeline..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/run_training.py

query-model:
	@echo "Running top-10 breakout prediction inference..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/query_model.py $(SEASON)

individual-data-query:
	@echo "Running individual player data query..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/individual_data_query.py $(PLAYER)

test:
	@echo "Running tests..."
	PYTHONPATH=$(PYTHONPATH) pytest tests/

lint:
	@echo "Checking code style with flake8..."
	flake8 src/ tests/

format:
	@echo "Auto-formatting with black..."
	black src/ tests/