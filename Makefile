# Data layer build.
#
#   make data   rebuild data/pdm.db from data/raw/, from scratch, deterministically
#   make test   run the test suite
#
# PYTHON defaults to whatever `python` resolves to, so activate the virtualenv
# first, or override it:  make data PYTHON=venv/Scripts/python.exe
#
# `make` is not installed with Git for Windows. Either install it
# (`winget install ezwinports.make`) or run the commands under each target by
# hand -- they are ordinary one-liners for exactly that reason.

PYTHON ?= python

RAW        := data/raw
DB         := data/pdm.db
GENERATED  := data/generated
INVENTORY  := $(GENERATED)/parts_inventory.csv
MANIFEST   := $(GENERATED)/build_manifest.json

RAW_FILES := $(RAW)/PdM_telemetry.csv $(RAW)/PdM_errors.csv $(RAW)/PdM_maint.csv \
             $(RAW)/PdM_failures.csv $(RAW)/PdM_machines.csv

.PHONY: all data inventory test verify investigate clean distclean help

all: data

## data: rebuild the database from scratch (inventory first, then ingest)
data: $(DB)

$(DB): $(RAW_FILES) $(INVENTORY) src/data/ingest.py src/data/schemas.py
	$(PYTHON) -m src.data.ingest --raw $(RAW) --db $(DB) \
		--inventory $(INVENTORY) --manifest $(MANIFEST)

## inventory: regenerate the seeded synthetic parts inventory
inventory: $(INVENTORY)

$(INVENTORY): scripts/generate_inventory.py $(RAW)/PdM_maint.csv $(RAW)/PdM_machines.csv
	$(PYTHON) scripts/generate_inventory.py --raw $(RAW) --out $(INVENTORY)

## test: run the test suite
test:
	$(PYTHON) -m pytest

## verify: print a profile of the raw files without building anything
verify:
	$(PYTHON) scripts/verify_data.py $(RAW)

## investigate: reproduce the failure/maint gap analysis in docs/DATA.md 5.1
investigate:
	$(PYTHON) scripts/investigate_failure_maint_gap.py $(RAW)

## clean: remove build outputs, keeping data/raw
clean:
	rm -f $(DB) $(INVENTORY) $(MANIFEST)
	rm -rf .pytest_cache
	find . -path ./venv -prune -o -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

## distclean: clean, and drop the generated directory entirely
distclean: clean
	rm -rf $(GENERATED)

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
