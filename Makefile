# Windmill -- pipeline runner
#
# The numbered scripts in scripts/pipeline/ have a dependency order that is
# easy to forget and easy to get wrong: 09_report.py reads every result file,
# so it must run last; 02_condition.py must precede 03_detect.py, and so on.
# This Makefile is the memory of that order.
#
# Every target carries a "## description" on its own line, which is what
# `make help` prints. This header names all of them, in five groups, and a
# test (tests/test_repo_rules.py) fails if a target is added without being
# listed here.
#
# Setup
#   make help            list every target with its description
#   make setup           create the venv (Python 3.11) and install dependencies
#   make data            download + checksum-verify the pinned datasets (19 GB, once)
#
# The pipeline: writes results/*.json. The order matters, so use the target.
#   make pipeline        run every artifact-producing step in dependency order;
#                        it finishes with `make report` and `make summary`
#   make report          regenerate README.md from results/ (after any change)
#   make summary         write results/summary.json, one table across every arm
#
# Evidence that is NOT part of `make pipeline` (slower; run on demand, and run
# `make report` afterwards so the README picks the result up)
#   make backtest        the evaluation suite: every claim, measured (about 90 s)
#   make daily           the daily check: a frozen monthly model predicts the healthy
#                        sensor value each day, alarms when the sensor departs from
#                        it; every sensor, a 7-day forecast and the sibling
#                        comparison; writes the dashboard's daily series
#                        (about 6 min)
#   make care-a          CARE Farm A against EDP's own records: true dates,
#                        integrity, and triage of the turbines' temperature alarms
#                        into real excess heat, load or weather (about 2 min)
#   make engie-triage    the same triage of ENGIE's fixed 80 C limit alarms
#   make care-bc         the daily check on CARE Farms B and C, monthly vs frozen
#   make alarm-value     EDP: the weather term, and alarm lead time vs standstill after failures
#   make confirm         the historical confirmation: train on each month, predict
#                        the next, and check it against what happened, over every
#                        month of every turbine (about 5 to 10 min)
#   make curve           the operating curve: false call-outs per year against
#                        time to catch a fault (needs LightGBM; about 2 min)
#   make cost            price the operating curve under stated assumptions
#                        (reads results/operating_curve.json, so run it after
#                        `make curve`; takes a moment)
#   make audit           re-run the autoencoder-comparability audit on all 95
#                        CARE datasets (about 1 h), then compare + thermal-signal
#   make audit-quick     the same audit on a small subset (a few minutes)
#   make compare         score both arms on the SAME CARE subset (like-for-like)
#   make thermal-signal  how much thermal signal is inside a fault window at all
#
# See it work
#   make demo            watch the detector work on real SCADA
#   make snapshot        export results/demo_r80711.json + assets/sample_turbine.csv
#   make dashboard       run the Streamlit dashboard (http://localhost:8501)
#
# Checks and housekeeping
#   make test            run the test suite
#   make diagnostics     run the one-off diagnostic probes (they write no artefacts)
#   make lint            syntax-check every Python file
#   make clean           remove caches and build artefacts (keeps results/, data/)
#   make clean-results   delete the generated result files (a deliberate act:
#                        they are committed)
#
# Rule 1 of the repo: results/*.json is the single source of truth and
# README.md is generated from it. If you change anything under src/windmill/,
# re-run the affected pipeline steps and then `make report`.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
SCRIPTS := scripts/pipeline

.DEFAULT_GOAL := help
.PHONY: help setup data pipeline report summary test demo audit audit-quick \
        compare thermal-signal dashboard snapshot diagnostics lint clean clean-results \
        backtest curve cost confirm daily care-a engie-triage care-bc alarm-value

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

backtest: ## run the evaluation suite: every claim, measured (results/backtest.json)
	$(PY) $(SCRIPTS)/15_backtest.py

daily: ## the daily check, every sensor, the 7-day forecast and the siblings (results/daily_check.json, results/daily_series.json)
	$(PY) $(SCRIPTS)/19_daily_check.py

care-a: ## CARE Farm A vs EDP's logbook and alarm log: integrity and alarm triage (results/care_farm_a.json)
	$(PY) $(SCRIPTS)/20_care_farm_a.py

engie-triage: ## ENGIE's 80 C limit alarms split into real excess heat, load or weather (results/engie_triage.json)
	$(PY) $(SCRIPTS)/21_engie_triage.py

alarm-value: ## EDP: error without the weather term; confirmed-heat lead time and standstill after each failure (results/alarm_value.json)
	$(PY) $(SCRIPTS)/23_alarm_value.py

care-bc: ## the daily check on CARE Farms B and C: false call-outs, monthly vs frozen, and faults caught (results/care_farms_bc.json)
	$(PY) $(SCRIPTS)/22_care_farms_bc.py

confirm: ## train on last month, predict this month, check it: every month of every turbine (results/monthly_confirmation.json)
	$(PY) $(SCRIPTS)/18_monthly_confirmation.py

curve: ## the operating curve: false call-outs per year against time to catch a fault (results/operating_curve.json)
	$(PY) $(SCRIPTS)/16_operating_curve.py

cost: ## price the operating curve under stated assumptions (results/cost_model.json)
	$(PY) $(SCRIPTS)/17_cost_model.py

demo: ## watch the detector work on real SCADA
	$(PY) $(SCRIPTS)/11_demo.py

snapshot: ## export results/demo_r80711.json + assets/sample_turbine.csv
	$(PY) $(SCRIPTS)/11_demo.py --export

dashboard: ## run the Streamlit dashboard (http://localhost:8501)
	$(PY) -m streamlit run app.py

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
