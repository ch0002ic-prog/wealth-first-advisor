PYTHON ?= .venv/bin/python
PYTHONPATH_VALUE ?= src

.PHONY: repro-suite repro-gate research-audit

repro-suite:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/run_main3_repro_suite.py

repro-gate: repro-suite
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/check_repro_drift.py

research-audit:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/run_optimization_research_audit.py
