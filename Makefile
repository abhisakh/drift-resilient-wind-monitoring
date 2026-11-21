# Windmill -- pipeline runner
#
# The numbered scripts in scripts/pipeline/ have a dependency order that is
# easy to forget and easy to get wrong: 09_report.py reads every result file,
# so it must run last; 02_condition.py must precede 03_detect.py, and so on.
# This Makefile is the memory of that order.
#
#   make help      list the targets
#   make setup     create the venv and install dependencies
#   make data      download + verify the pinned datasets (19 GB, once)
#   make pipeline  run every artifact-producing step, in order
#   make report    regenerate README.md from results/ (after any change)
#   make test      run the test suite
#   make demo      watch the detector work on real SCADA
#
# Rule 1 of the repo: results/*.json is the single source of truth and
# README.md is generated from it. If you change anything under src/windmill/,
# re-run the affected pipeline steps and then `make report`.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
SCRIPTS := scripts/pipeline

.DEFAULT_GOAL := help
.PHONY: help setup data pipeline report summary test demo audit audit-quick \
        compare thermal-signal diagnostics lint clean clean-results

help: ## list the available targets
	@echo ""
	@echo "Windmill -- pipeline targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Typical use:  make setup && make data && make pipeline"
	@echo ""

setup: ## create the venv (Python 3.11) and install dependencies
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data: ## download + checksum-verify the pinned public datasets (19 GB)
	$(PY) $(SCRIPTS)/01_build_dataset.py

# --- the pipeline, in dependency order --------------------------------------
# 02* before 03* (conditioning feeds detection); 08b/08c/12 are independent
# of each other; 09 and 10 read everything and must come last.
pipeline: ## run every artifact-producing step, in order
	$(PY) $(SCRIPTS)/02_condition.py
	$(PY) $(SCRIPTS)/02b_decompose.py
	$(PY) $(SCRIPTS)/02c_per_turbine_curve.py
	$(PY) $(SCRIPTS)/03_detect.py
	$(PY) $(SCRIPTS)/03b_residual_attribution.py
	$(PY) $(SCRIPTS)/04_score.py
	$(PY) $(SCRIPTS)/05_audit_scorability.py
	$(PY) $(SCRIPTS)/07_lightgbm_nbm.py
	$(PY) $(SCRIPTS)/08_multivariate.py
	$(PY) $(SCRIPTS)/08b_sweep_threshold.py
	$(PY) $(SCRIPTS)/08c_engie_multivariate.py
	$(MAKE) report
	$(MAKE) summary

report: ## regenerate README.md from results/ (single source of truth)
	$(PY) $(SCRIPTS)/09_report.py

summary: ## write results/summary.json -- one table across every arm
	$(PY) $(SCRIPTS)/10_summary.py

audit: ## re-run the AE-comparability audit (all 95 datasets, ~1 h)
	$(PY) $(SCRIPTS)/12_ae_audit.py --datasets 0 --configs published:0,no_status:0,published:1
	$(MAKE) compare
	$(MAKE) thermal-signal

audit-quick: ## AE-comparability audit on a small subset (a few minutes)
	$(PY) $(SCRIPTS)/12_ae_audit.py --datasets 2 --configs published:0,no_status:0
	$(MAKE) compare

compare: ## score both arms on the SAME CARE subset (like-for-like vs the AE)
	$(PY) $(SCRIPTS)/13_subset_comparison.py --datasets 4

thermal-signal: ## how much thermal signal is inside a fault window at all
	$(PY) $(SCRIPTS)/14_thermal_signal.py

demo: ## watch the detector work on real SCADA
	$(PY) $(SCRIPTS)/11_demo.py

test: ## run the test suite
	$(PY) -m pytest tests/ -q

diagnostics: ## run the one-off diagnostic probes (they write no artefacts)
	$(PY) scripts/diagnostics/07a_check_determinism.py

lint: ## syntax-check every Python file
	$(PY) -m compileall -q src scripts tests

clean: ## remove caches and build artefacts (keeps results/ and data/)
	rm -rf .pytest_cache
	find . -name "__pycache__" -type d -not -path "./.venv/*" -exec rm -rf {} +
	find . -name "*.pyc" -not -path "./.venv/*" -delete

clean-results: ## delete the generated result files (they are committed, so this is a deliberate act)
	rm -f results/*.json
