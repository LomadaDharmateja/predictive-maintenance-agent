# RESULTS_SUMMARY.md

Every headline number, with the caveat that qualifies it. One page.

Source data is Microsoft's Azure predictive-maintenance **teaching simulation**,
not a plant, and the parts inventory is **synthetic**. Nothing below supports a
claim about real downtime or real cost.

---

## The headline: the warning is shorter than the supply chain

| | |
|---|---|
| Parts orderable inside the model's warning | **1 of 9** |
| Parts clearing it with the 1.25 safety factor | **0 of 9** |
| Detection lead, comp1 | 24.0 h (5 of 9 events detected) |
| Detection lead, comp2 / comp3 / comp4 | 335.0 h / 326.5 h / 335.0 h |
| Supplier lead times | 10–34 days, median 23 |

**Caveat.** The *structure* is real — a warning shorter than a supply chain. The
specific "1 of 9" depends on invented lead times. On a real inventory the ratio
would differ; the method for finding it would not.

**Why it cannot be fixed by predicting further ahead.** Predictability caps the
horizon at 14 days (past that the model's interval overlaps a matched-error-code
baseline). The lead-time requirement starts at 23 days. The two constraints do
not intersect.

**Consequence.** Risk prediction schedules maintenance attention. Parts
management runs off stock and consumption and never sees a prediction — enforced
by an import-graph test, not by convention.

---

## Model performance — test split, opened once, 14-day horizon

Logistic regression. PR-AUC with 95% bootstrap intervals resampled at the
**failure-event** level, because rows are not independent.

| Component | PR-AUC | 95% CI | Matched-code baseline | Majority |
|---|---:|---|---:|---:|
| comp1 | 0.1902 | [0.1308, 0.2494] | 0.0686 | 0.0537 |
| comp2 | 0.2529 | [0.1941, 0.3135] | 0.1408 | 0.1145 |
| comp3 | 0.3773 | [0.2620, 0.4940] | 0.0599 | 0.0465 |
| comp4 | 0.2481 | [0.1702, 0.3230] | 0.0961 | 0.0577 |

**Caveats.** Positives derive from **121 distinct failure events** — the
intervals are wide because the evidence is thin, and the interval is the result,
not the point estimate. Accuracy is never reported: the positive rate makes it
meaningless. Logistic regression ships over LightGBM because the paired
bootstrap on their difference **spans zero** — neither is established as better,
and the simpler one is 464× smaller and readable.

**Ignore the 1.000.** An earlier 24-hour framing scored perfect PR-AUC. It was
real, and useless: a 24-hour warning cannot serve a 23-day lead time. Archived in
`docs/EVALUATION_24h.md`.

---

## Calibration — one probability of four is trustworthy

Held-out Brier skill after isotonic calibration, cross-fitted over five
contiguous time blocks.

| Component | Skill | 95% CI | Trustworthy |
|---|---:|---|---|
| comp1 | +0.0049 | (−0.1041, +0.0251) | no |
| comp2 | **+0.0667** | **(+0.0227, +0.0945)** | **yes** |
| comp3 | +0.1540 | (−0.4086, +0.2650) | no |
| comp4 | +0.1203 | (−0.0318, +0.1983) | no |

**Only comp2's interval excludes zero.** The other three carry
`calibrated: false` in the tool output and the agent must say so in the same
sentence as the number.

**The collision worth knowing.** comp2 is the only component you can *believe*.
comp4 is the only one whose warning is long enough to *act on*. They are
different components.

---

## Agent evaluation — 41 scenarios × 3 seeds

Agent `claude-sonnet-5`, judge `claude-haiku-4-5`, 123 recorded runs, **$2.71**.

| | |
|---|---|
| Scenarios passed | **47 / 123** |
| Forbidden tool calls | **0** — the design separation held across every run |
| Runs with an ungrounded figure | 16 |
| Deterministic replay | **123 / 123 identical**, no model called |

**The pass rate is not an agent-quality score**, for two measured reasons:

1. **The judge is not calibrated.** κ = **0.602** against a 0.7 floor (48 blind
   hand labels). Every judged assertion is an unvalidated opinion. 11 of 15
   assertion types agreed perfectly — but 9 of those rest on 1–2 rows.
2. **Most grounding failures are the harness, not the agent.** All 16 distinct
   flagged figures were hand-inspected. **None** was invented by the agent: they
   are arithmetic the grounding check deliberately excludes, plus two known
   tokeniser defects.

---

## The retracted finding

`obeys_injected_instruction` fired 6 times and was initially reported as the
agent obeying injected instructions. **Retracted.** Its positively-framed twin
`injection_not_obeyed` failed **zero** times across all 12 injection runs and
agrees with human labels 4/4. The agent appears to have resisted the injections;
the judge inverted on the negatively-framed name.

Without measuring judge agreement, this project would have shipped a false
security finding as a headline.

---

## Engineering

| | |
|---|---|
| Tests | **519**, no network required |
| Deterministic builds | Two clean builds produce identical content hashes |
| Test split | Opened once per horizon, behind a single-consumer token |
| Prompt caching | 75.5% of input tokens served from cache; input spend down 67.8% |
| Container | Non-root uid 10001; no compiler, no package installer, no secret or database inside — **verified by inspecting the built image** |

---

## What this does not tell you

- Nothing here has touched real plant data or a real maintenance planner.
- Three seeds are three recorded occasions, not a resampling of a stochastic
  system.
- Data-channel prompt injection is **not tested** — no tool returns a free-text
  field an attacker could write into, so the four scenarios test the user
  channel and say so.
- Whether a planner behaves differently given `warning_adequacy: insufficient`
  is untested. An agent that is correct and ignored has failed.

Full detail: [`../README.md`](../README.md),
[`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md),
[`AGENT_EVALUATION.md`](AGENT_EVALUATION.md).
