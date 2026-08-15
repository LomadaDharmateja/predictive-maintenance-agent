# CLAUDE.md

Project instructions loaded into every session. Keep this current.

## What this project is

A predictive-maintenance agent over the Microsoft Azure PdM dataset:
100 machines, 876,100 hourly telemetry rows, plus error, maintenance
and failure logs. An LLM agent answers maintenance-planning questions
through typed tools over a calibrated risk model, maintenance history
and a synthetic parts inventory.

This is a rebuild. An earlier version (VULCAN) was audited, found to
be built on three unrelated datasets, and replaced. That version is in
`archive/` and its audit is in `docs/v1/`. Do not treat anything in
`archive/` or `docs/v1/` as current.

## Problem statement

Flag elevated component risk over a 14-day window so maintenance
attention can be scheduled, and manage parts from stock levels and
consumption rates rather than from predictions.

The second clause is a measured constraint, not a preference. No
prediction horizon satisfies both model accuracy and the 23-day median
parts lead time; one of nine parts can be ordered in time. See
`docs/EVALUATION.md`.

## Read these before working

- `docs/DATA.md` — schema, leakage rules, cost assumption
- `docs/FEATURES.md` — every feature and its window
- `docs/EVALUATION.md` — model results and what they do not support
- `docs/MILESTONE_*.md` — the spec for the milestone in progress

## Standing rules

**Temporal integrity.** No feature may use a record with
`datetime > t`. Splits are temporal with an embargo derived from the
label horizon. No random or shuffled splitting anywhere.

**The test split is opened once**, at the end of a milestone, after
every modelling decision is final. One module may load it. Choosing
between trained models by their test scores is a modelling decision
taken on test.

**Metrics.** PR-AUC with bootstrap confidence intervals, against the
majority and rule-based baselines. Accuracy is never reported; the
positive rate is under 1%.

**Calibration.** A probability is presented as trustworthy only when
its held-out Brier skill is positive and its confidence interval
excludes zero. Otherwise it carries `calibrated: false` and the agent
must surface that.

**Database access.** The model never writes SQL. Typed filters only,
read-only connections, authorizer allowlist. Verified by content hash.

**Errors.** Tool results are `Success[T] | ToolError` as distinct
types. A failed call must never be presentable as a successful one.
No bare `except Exception` that returns a value.

**Determinism.** Seed everything. Two clean builds produce identical
content hashes. No clock, no network, no environment in the pipeline.

## Working style

- Report what you measured, not what you expect. Mark anything
  unverified as `UNVERIFIED:`.
- If a spec I wrote is wrong, say so and say why rather than
  implementing it.
- Negative results are results. Report them plainly.
- Stop at the stated milestone boundary. Do not proceed to the next.
- Do not print or copy any credential value.

## Environment

Windows, PowerShell, venv at `venv/`. `make` is not installed —
run recipe steps directly.
