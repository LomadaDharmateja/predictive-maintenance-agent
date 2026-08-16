# How this system works

A walkthrough for a reader who wants to understand the system end to end without
reading the code. Every number here is measured; where a number is uncertain,
the uncertainty is given alongside it.

---

## 1. What it is

An LLM agent that answers maintenance-planning questions over a 100-machine
fleet. A planner asks something in plain language — *"What's the comp2 position
on machine 96 at the moment?"* — and the agent decides which of six typed tools
to call, calls them, and writes an answer from what they return.

The six tools:

| Tool | Returns |
|---|---|
| `get_failure_risk` | A calibrated 14-day failure probability, its confidence interval, whether it is trustworthy, and whether the warning is long enough to act on |
| `get_parts_position` | Stock on hand, observed consumption rate, and how long stock would last |
| `get_maintenance_history` | Past component replacements for a machine |
| `get_recent_errors` | Non-fatal error codes in a recent window |
| `get_machine_profile` | Model and age |
| `find_machines` | Fleet-wide search by typed filters |

The agent never writes SQL. `find_machines` takes a typed filter object and
builds the query itself; connections open read-only behind an authorizer
allowlist. A hallucinated `machine_id` of 250 fails validation and becomes a
typed error before it reaches the database.

---

## 2. The data, and why it suits this problem

Microsoft's Azure predictive-maintenance sample: five CSVs covering one year of
a 100-machine fleet.

| Table | Rows | What it is |
|---|---:|---|
| `telemetry` | 876,100 | Hourly volt, rotate, pressure, vibration per machine |
| `errors` | 3,919 | Non-fatal error codes |
| `maint` | 3,286 | Component replacements |
| `failures` | 761 | Component failures |
| `machines` | 100 | Model and age |
| `parts_inventory` | 9 | Synthetic, generated from a seed |

Four properties make it the right shape for this problem:

- **The tables join.** Every table keys on `machineID` and an hourly
  `datetime`, so telemetry, errors, maintenance and failures describe the same
  machines on the same clock. A question like *"was there an error before this
  failure?"* is answerable rather than assumed.
- **It is hourly and continuous for a full year.** That supports time-windowed
  features — rolling means and standard deviations over 3 h and 24 h, error
  counts over 24 h and 7 d, component age since last replacement — and it
  supports temporal splitting, which a snapshot dataset would not.
- **Failures are labelled and attributed to a component.** 761 failure events,
  each naming which of four components failed, so the model can be fitted per
  component rather than as one undifferentiated "something broke" signal.
- **It is realistically imbalanced.** Failures are rare against 876,100 rows.
  That forces the honest metrics this project uses — PR-AUC against baselines,
  never accuracy, which would read as 99% for a model that predicted nothing.

Its limits are equally worth stating, because they bound every result:

- **It is a teaching simulation, not a plant.** The fault signatures are
  injected and clean in a way real telemetry is not. Nothing measured here
  supports a claim about real downtime, real cost, or real equipment.
- **The parts inventory is invented.** Lead times, stock levels and unit costs
  are generated from a seed. The lead-time finding in section 4 is structurally
  real; its specific ratio depends on numbers this project made up.
- **The evaluation set is small.** Positives on the test split derive from 121
  distinct failure events, which is why every interval in section 5 is
  bootstrapped at the failure-event level rather than the row level.

---

## 3. From raw rows to a probability

The offline pipeline runs in four deterministic stages. Two clean builds produce
identical content hashes; nothing reads the clock, the network or the
environment.

1. **Load.** The five CSVs are checksummed and loaded into SQLite.
2. **Featurise.** At each prediction time `t`, for each machine, 38 features are
   computed — rolling telemetry statistics, error counts, component age, machine
   model and age. Every window is half-open `(t-W, t]`: **no feature may use a
   record stamped later than `t`.** This matters most for `maint`, where a
   replacement record at or after `t` would leak the outcome directly. The rule
   is enforced by a test that parses the source, and that scanner is itself
   tested against a planted violation so it cannot pass by being broken.
3. **Split, temporally.** Train is January–September, validation October, test
   November–December, with a 14-day embargo between them derived from the label
   horizon. No random or shuffled split exists anywhere. The test split opens
   once, at the end of a milestone, behind a single-consumer token.
4. **Fit and calibrate.** One logistic regression per component, then isotonic
   calibration cross-fitted over five contiguous time blocks so the calibrator
   never sees a row's own neighbours.

The label: *does this component fail within the next 14 days?* Section 4
explains where that 14 came from.

---

## 4. Why parts are managed from stock, not from predictions

This is the decision that shapes the architecture, so it is worth following
carefully.

The obvious design is predict-and-order: forecast a failure, order the part,
fit the replacement before it breaks. Whether that works depends on two numbers.

The first is the model's **effective detection lead** — not how far ahead the
label looks, but how long before the failure the score actually crosses its
operating threshold. The second is the **supplier lead time** for the part.

| Component | Detection lead (median) | Events detected | Shortest part lead time |
|---|---:|---:|---:|
| comp1 | **24.0 h** | 5 of 9 | 240 h (10 d) |
| comp2 | 335.0 h | 27 of 27 | 768 h (32 d) |
| comp3 | 326.5 h | 8 of 8 | 360 h (15 d) |
| comp4 | 335.0 h | 14 of 15 | 288 h (12 d) |

Supplier lead times across the nine stocked parts run 10 to 34 days, median 23.
The model's reliable warning horizon is 14 days. Crossing the two lists:

> **1 of 9 parts can be ordered inside the warning the model gives.**
> **0 of 9 clear it with the 1.25 safety factor applied.**

The single part that fits — `PN-COMP4-001`, 335 h of warning against a 288 h
lead time — clears the lead time and fails the safety factor. Its verdict is
`marginal`. There is no `sufficient` pair in the inventory.

**Neither a longer nor a shorter horizon rescues it.** Predictability caps the
horizon at 14 days: past that, the model's bootstrap interval overlaps a
matched-error-code baseline's, and it is no longer established as better than a
spreadsheet rule. The lead-time requirement starts at 23 days. The two
constraints do not intersect. Going the other way, a 24-hour horizon scores a
genuine test PR-AUC of 1.000 on two components — the simulator's error-code and
sensor signature is near-deterministic that close in — and is useless, because a
24-hour warning cannot inform a decision whose action takes three weeks.

So the system was designed around what the measurement supports:

> Flag elevated component risk over a 14-day window so maintenance attention can
> be scheduled, and manage parts from stock levels and consumption rates rather
> than from predictions.

Risk prediction schedules a technician's attention, which is a decision that fits
inside 14 days. Parts management runs off stock on hand and observed consumption,
which needs no prediction at all. **The two paths never meet, and that is
enforced rather than intended:** `get_parts_position` accepts no risk score and
has no import path to the model, asserted by a test that walks the import graph.
A future change that wires risk into parts reasoning fails the build.

The agent also reports this per question. When a planner asks whether to order a
part, `warning_adequacy` tells them the warning is `insufficient` and the agent
declines to recommend the order — rather than producing a confident
recommendation the supply chain cannot execute.

---

## 5. How well the model actually works

Test split, opened once, at the 14-day horizon. PR-AUC with 95% bootstrap
intervals resampled at the failure-event level, because rows are not
independent — one failure positively labels up to 336 consecutive prediction
times for that machine.

| Component | Model PR-AUC | 95% CI | Matched-code baseline | Majority baseline |
|---|---:|---|---:|---:|
| comp1 | 0.1902 | [0.1308, 0.2494] | 0.0686 | 0.0537 |
| comp2 | 0.2529 | [0.1941, 0.3135] | 0.1408 | 0.1145 |
| comp3 | 0.3773 | [0.2620, 0.4940] | 0.0599 | 0.0465 |
| comp4 | 0.2481 | [0.1702, 0.3230] | 0.0961 | 0.0577 |

The model beats both baselines on every component. The intervals are wide
because 121 failure events is not many, and the interval is the result — not the
point estimate.

Logistic regression ships rather than LightGBM. The paired bootstrap on their
PR-AUC difference spans zero on all four components, as does the paired
difference in calibrated Brier skill, so neither is established as better.
Logistic regression is 464× smaller and its coefficients can be read.

### Which probabilities can be believed

Being able to rank machines is not the same as being able to quote a number.
Held-out Brier skill after calibration:

| Component | Skill (held out) | 95% CI | Trustworthy |
|---|---:|---|---|
| comp1 | +0.0049 | (−0.1041, +0.0251) | no |
| comp2 | **+0.0667** | **(+0.0227, +0.0945)** | **yes** |
| comp3 | +0.1540 | (−0.4086, +0.2650) | no |
| comp4 | +0.1203 | (−0.0318, +0.1983) | no |

Only comp2's interval excludes zero. The other three are not established as
better than simply reporting the base rate, so they carry `calibrated: false`
in the tool output, and the agent is required to say so in the same sentence as
the number rather than in a footnote.

There is a collision worth naming: **comp2 is the only component you can
believe, and comp4 is the only one whose warning is long enough to act on.**
They are different components. The probability you can trust is not the one you
can use.

---

## 6. How the agent is evaluated

41 hand-written scenarios covering lookups, risk questions with adequate and
inadequate warning, parts positions, multi-step questions, tool failures,
unanswerable questions, and prompt injection. Each runs at 3 seeds — 123 runs
in total, recorded as transcripts so the suite replays offline, deterministically
and free.

Each run is checked on four axes: did it call the right tools and no forbidden
ones; is every figure in the answer traceable to something a tool returned; does
it satisfy the scenario's assertions; and did it avoid inventing anything.

| Category | Passed |
|---|---|
| tool_failure | 12/15 |
| unanswerable | 10/12 |
| risk_inadequate_warning | 9/24 |
| lookup | 8/15 |
| prompt_injection | 4/12 |
| risk_adequate_warning | 2/15 |
| multi_step | 1/15 |
| parts_position | 1/15 |
| **Total** | **47/123** |

**Zero forbidden tool calls across all 123 runs.** `get_failure_risk` was never
called inside a parts question — the separation in section 4 held in practice,
not just in the type system.

**The 47/123 should not be read as an agent-quality score.** Assertions are
graded by a second model, and that judge was checked against 48 blind human
labels: Cohen's κ = 0.602, against a 0.7 floor. It did not clear the bar, so
every judged assertion is an unvalidated opinion and is marked as such.
Separately, all 16 distinct figures flagged as ungrounded were hand-inspected
and **none** was invented by the agent — they are arithmetic the grounding check
deliberately excludes, plus two known tokeniser defects. The pass rate measures
the harness at least as much as the agent.

That calibration exercise earned its cost once already. An assertion named
`obeys_injected_instruction` fired 6 times and was initially reported as the
agent obeying injected instructions — a headline security finding. Its
positively-framed twin `injection_not_obeyed` failed zero times across all 12
injection runs and agreed with human labels 4 out of 4. The agent had resisted
the injections; the judge had inverted on the negatively-framed name. The
finding was retracted.

---

## 7. What runs where

The agent loop is written rather than adopted from a framework: bounded
iterations with a defined terminal behaviour, a bounded transcript whose drops
are recorded, and tool results typed as `Success[T] | ToolError` so a failed
call cannot be presented as a successful one.

One `LLMClient` protocol sits behind both a hosted Anthropic adapter and a local
Ollama adapter, selected by configuration, so the question of whether data may
leave the building is answerable before a pilot rather than after. API keys come
from the environment and are never read, logged or printed by this repository.

Every run emits OpenTelemetry spans — `agent.run` over `model.call` and
`tool.call` — from which token counts, cost and iteration counts are computed.
A recorded run can be replayed offline and checked for divergence, or rendered
as a self-contained HTML trace.

The service is FastAPI (`/v1/ask`, `/v1/runs/{id}`, `/health`) in a multi-stage
container that runs as a non-root user with no compiler, no package installer,
no secret and no database inside — verified by inspecting the built image rather
than by trusting the Dockerfile. 519 tests run without network access.

---

## 8. What this does not tell you

- Nothing here has touched real plant data or a real maintenance planner.
- Three seeds are three recorded occasions, not a resampling of a stochastic
  system.
- Prompt-injection coverage is narrower than the category name suggests. No tool
  returns a free-text field an attacker could write into, so the four scenarios
  test the user channel and say so. See [`SECURITY.md`](SECURITY.md).
- Whether a planner behaves differently when told `warning_adequacy:
  insufficient` is untested. An agent that is correct and ignored has failed.

Fuller detail: [`../README.md`](../README.md),
[`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md),
[`AGENT_EVALUATION.md`](AGENT_EVALUATION.md). The decisions behind the design,
each with the measurement that drove it, are in
[`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md).
