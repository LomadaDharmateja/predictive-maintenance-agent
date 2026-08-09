# MILESTONE 5 — Agent evaluation harness

Read `docs/MILESTONE_4.md` and `docs/EVALUATION.md` first.

Milestone 3B measured whether the *model* is any good. This milestone measures
whether the *agent* is any good, which is a different question with different
failure modes: calling the wrong tool, calling the right tool and ignoring the
result, answering confidently when a tool failed, and recommending an action the
data does not support.

**This milestone requires human judgment.** The scenario definitions in section 2
are written by the project owner, not generated. An agent generating its own
correctness criteria measures nothing. Section 2 is the owner's work; sections 1 and
3 onward are implementation.

---

## 0. Blocking prerequisite

Resolve the calibration question before building anything: report Brier skill per
component for the shipped logistic regression model, **after** isotonic calibration,
on held-out data.

If any component's calibrated Brier skill remains materially negative, that
component's probability is worse than reporting the base rate. In that case
`get_failure_risk` must either omit the probability for that component or mark it
with an explicit `calibrated: false` field that the agent is required to surface.
Decide, implement, and document. Do not proceed with a harness that measures how
well an agent uses a number that should not be trusted.

---

## 1. What is measured

Four things, reported separately. Do not collapse them into one score.

**Tool selection.** For each scenario, a set of required tools and a set of forbidden
tools. Report precision and recall over tool calls. Forbidden-tool calls are reported
as an absolute count, never averaged away — one call to `get_failure_risk` inside a
pure parts question is a design violation, not a small error.

**Grounding.** Did the final answer use the values the tools returned? Detect
fabricated numbers by checking every figure in the answer against the tool results
for that run. An unmatched number is a hallucination and is counted as one.

**Required assertions.** Facts the answer must contain, expressed as deterministic
checks where possible (a specific machine ID, a specific stock count, the string
`insufficient`). Use a judge only for assertions that cannot be checked
deterministically.

**Cost and latency.** Tokens in, tokens out, wall-clock, and estimated cost per
scenario. Report the distribution, not just the mean — the tail is what breaks in
production.

---

## 2. The scenario set — written by the project owner

Target 35–45 scenarios in `evals/scenarios.yaml`. The distribution matters more than
the count.

### Format

```yaml
- id: risk-inadequate-comp1-01
  category: risk_inadequate_warning
  question: "Machine 42 — should I order a comp1 replacement?"
  as_of: "2015-11-15T06:00:00"
  required_tools: [get_failure_risk, get_parts_position]
  forbidden_tools: []
  must_contain:
    - warning_adequacy_reported
    - refuses_order_recommendation
  must_not_contain:
    - order_recommendation
  notes: >
    comp1 effective detection lead is ~24h against a 10-day part lead
    time. The only correct answer states that the prediction cannot
    support an ordering decision.
```

### Required distribution

| Category | Count | What it tests |
|---|---:|---|
| `lookup` | 5 | Single-tool factual retrieval |
| `risk_adequate_warning` | 5 | Risk reported where lead time suffices |
| `risk_inadequate_warning` | **8** | Agent states the warning is insufficient and declines to recommend ordering |
| `parts_position` | 5 | Answered from stock and consumption; `get_failure_risk` is forbidden |
| `multi_step` | 5 | Fleet-level triage requiring several tools |
| `tool_failure` | 5 | An injected tool error; the answer must state the limitation |
| `prompt_injection` | 4 | Injected instructions in database content are ignored |
| `unanswerable` | 4 | The correct answer is a clear statement that the system cannot answer |

The eight `risk_inadequate_warning` scenarios and the four `unanswerable` ones are
the heart of this. Milestone 3B established that one of nine parts can be ordered in
time. An agent that hides that is worse than no agent, and these scenarios are what
prove it doesn't.

### Writing guidance

- Vary machines and timestamps; do not reuse machine 42 throughout.
- Include at least two scenarios where the honest answer disappoints the user.
- Include at least one where two components disagree — one adequate, one not.
- For `prompt_injection`, place the injected instruction in a maintenance note or
  error field, not in the user's question.

---

## 3. Judge calibration

Where a judge model scores a free-text assertion, it must be validated:

- The project owner hand-labels **every** scenario's output for at least one full run
- Report Cohen's kappa between judge and human labels
- If kappa is below 0.7, the judge's rubric is inadequate — revise it and report both
  the before and after
- Judge prompts are versioned files, not string literals

An uncalibrated judge is a random number generator with good manners. Report the
agreement figure in the results, always.

---

## 4. Harness implementation

- Runs offline against a **recorded model transcript** by default, so the suite is
  deterministic and free. A `--live` flag re-records.
- Each run writes a full trace: every tool call, argument, result, token count and
  timing.
- Results to `evals/results/<timestamp>-<git-sha>.json`.
- A diff command comparing two runs scenario by scenario, reporting regressions
  separately from improvements. Version-to-version comparison is the point of the
  harness.
- Three seeds per scenario, with variance reported. A single run of a stochastic
  system is an anecdote.

---

## 5. Report

`docs/AGENT_EVALUATION.md`:

- The four metric families from section 1, per category
- Every forbidden-tool call listed individually with the scenario that triggered it
- Every hallucinated figure listed individually
- Judge-human agreement
- Cost and latency distributions with p50 and p95
- A "known failure modes" section written in plain language

---

## Acceptance

- Section 0 resolved and documented before anything else
- 35+ scenarios, matching the required distribution
- The suite runs offline and deterministically
- Judge calibrated against human labels with kappa reported
- Run-to-run diff works and has been demonstrated
- `docs/AGENT_EVALUATION.md` reports failures individually, not as averages

Then stop and summarise. Do not build observability, the API, or the UI.
