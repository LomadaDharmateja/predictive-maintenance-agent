# ENGINEERING_NOTES.md

What was wrong, what was done about it, and the measurement in each case.

This is not a changelog. It is the set of decisions I would expect to defend
line by line, written the way I would explain them to another engineer who
asked "why is it like that?" Several of these are places where the honest
answer made the project look worse.

---

## 1. The v1 audit: three datasets that could not be joined

The predecessor (VULCAN) presented itself as a predictive-maintenance agent over
a factory's data. It was built on three CSVs with no key between them:

| File | Rows | What it actually was |
|---|---:|---|
| `maintenance.csv` | 10,000 | The AI4I 2020 synthetic milling-machine benchmark |
| `commodity.csv` | 49,093 | Commodity price series |
| `industrial.csv` | 2,342 | An unrelated logistics table |

No shared identifier existed. `maintenance.csv` used `UDI`/`Product ID` from a
public benchmark; `industrial.csv` keyed on supplier records; `commodity.csv` was
a time series with no machine dimension at all. The agent's tools queried them as
though they described one factory. A question like "which supplier serves the
machine that is failing" returned an answer assembled from two tables that had
never described the same object.

**What was done.** The whole thing was replaced rather than patched. The rebuild
uses one coherent source — Microsoft's Azure PdM sample — where `machineID`
genuinely joins telemetry, errors, maintenance and failures. The one synthetic
addition, the parts inventory, is generated *from* that dataset's components and
is labelled synthetic everywhere it appears, including in the headline finding.

**Why replacing rather than fixing.** There was no fix. The claims were not
supported by the data because the data could not support any joint claim.

---

## 2. The 22-point leakage gap

v1 reported an F1 of 0.958. Recomputed from the archived training code
(`scripts/leakage_case_study.py`, reproducible with `make case-study`):

| | F1 | Precision | Recall |
|---|---:|---:|---:|
| v1's method (duplicated rows, random split) | **0.958** | 0.973 | 0.944 |
| Same model, honest split | **0.736** | 0.855 | 0.646 |

**A 22.2-point gap, entirely mechanical.** The training script duplicated the
10,000-row source to 50,000 rows — 10,000 unique ids appearing five times each,
40,000 of them duplicates — and then split randomly. Every test row had four
copies in training. The model was being asked to recognise rows it had already
memorised, and 0.958 is what that scores.

**What was done in the rebuild.**

- **Temporal splits only.** No random or shuffled split exists anywhere in
  `src/`, and `tests/test_no_future_leakage.py` asserts that by parsing the
  source rather than by trusting a convention.
- **An embargo derived from the horizon in code**, not written as a separate
  constant, so changing the horizon cannot leave a stale gap. It costs
  validation fourteen of its thirty-one days: 408 prediction times, not 720.
- **The guard is itself tested.** A scanner that never fires proves nothing, so
  it is run against a planted violation to confirm it catches one.
- **The test split is opened once**, behind a single-consumer token, with nine
  tests around the lock — including one that plants a reader.

**What I would still call unresolved.** The test split has now been opened
twice: once at 24 hours, once at 14 days. Both openings are recorded in
`build_manifest.json` with their git commit and model fingerprint. I think a
horizon change is a new question rather than a second look at the same one, but
it is a judgement call and the record is there to disagree with.

---

## 3. The 24-hour horizon that could not serve a 23-day lead time

Milestone 3 scored **test PR-AUC 1.000** on comp2 and comp3. The controls say it
is neither leakage nor memorisation: the simulator injects a matched error code
and sensor signature that fires before 100% of failures, and the pair is
near-deterministic 24 hours out. It is a correct measurement.

It is also worthless, and noticing that was the most consequential thing in the
project.

The system exists so a planner can decide whether to reserve or order a part.
`parts_inventory.csv` records supplier lead times of **10 to 34 days, median 23**.
A 24-hour warning cannot inform a decision whose action takes three weeks. The
horizon had been inherited from the standard framing of this dataset rather than
derived from the decision it serves — and everything built on top of it,
calibration and thresholds and the cost curve, was vacuous as a result.

**What was measured.** Effective detection lead — how long before the failure
the score actually crosses its operating threshold, which is the number that
matters and is not the label horizon:

| Component | Detection lead | Events detected | Shortest part |
|---|---:|---:|---:|
| comp1 | 24.0 h | 5 of 9 | 240 h |
| comp2 | 335.0 h | 27 of 27 | 768 h |
| comp3 | 326.5 h | 8 of 8 | 360 h |
| comp4 | 335.0 h | 14 of 15 | 288 h |

**1 of 9 parts fits inside the warning. 0 of 9 clear the 1.25 safety factor.**

Extending the horizon does not rescue it. At 30 days — the shortest horizon
covering the median lead time — the model's bootstrap interval overlaps the
matched-code baseline's on comp1 and comp2, so predictability is no longer
established. Predictability caps at 14 days; the lead-time requirement starts at
23. **No horizon satisfies both.**

**What was done.** The product was split in two. Risk prediction schedules
attention. Parts management runs off stock and consumption and never sees a
prediction. That separation is in the type system: `get_parts_position` accepts
no risk score, and `tests/test_agent_parts_independence.py` walks the import
graph to prove it has no path to the model.

**The alternative I rejected.** Picking 30 days anyway and reporting the PR-AUC
without the interval. It would have looked better and been indefensible under
one question.

---

## 4. Three of four probabilities are not trustworthy

Before building an evaluation harness that measures how well an agent uses a
number, it is worth checking the number is worth using.

Isotonic calibration fitted on all of validation and measured on the same rows
is in-sample and flatters itself. This is cross-fitted: validation is cut into
five contiguous time blocks and each is scored by a calibrator fitted on the
other four. Blocks are contiguous rather than random because adjacent hours are
near-identical, and a random fold would let the calibrator see a row's own
neighbours.

| Component | Base rate | Skill, raw | Skill, calibrated (held out) | 95% CI | Trusted |
|---|---:|---:|---:|---|---|
| comp1 | 0.0254 | −0.1160 | +0.0049 | (−0.1041, +0.0251) | no |
| comp2 | 0.1491 | −0.7640 | **+0.0667** | **(+0.0227, +0.0945)** | **yes** |
| comp3 | 0.0312 | +0.1408 | +0.1540 | (−0.4086, +0.2650) | no |
| comp4 | 0.0647 | +0.1148 | +0.1203 | (−0.0318, +0.1983) | no |

Isotonic repairs the sign — raw skill was negative on comp1 and comp2, and after
calibration nothing is materially negative. But **only comp2's interval excludes
zero.** comp1's point estimate of +0.005 is indistinguishable from no skill at
all. comp3's interval runs from −0.409 to +0.265, which is a statement about how
little validation data sits behind it rather than a claim of quality.

**What was done.** `ComponentRisk` carries `calibrated`, the held-out skill and
its interval as required fields. Where `calibrated` is false the agent must say
so in the same sentence as the number. The alternative — omitting the
probability — was considered and rejected: nothing is *materially negative*, so
the ranking signal is still useful, and hiding it would have removed information
the planner can legitimately use for triage.

**The finding underneath the finding.** comp2 is the only component you can
believe. comp4 is the only one whose warning is long enough to act on. They are
different components. That collision is more interesting than either number
alone, and I would rather report it than average it away.

---

## 5. Judge calibration: κ = 0.602, and a retracted finding

Fifteen of the assertions in the evaluation suite cannot be checked
deterministically and go to an LLM judge. An uncalibrated judge is a random
number generator with good manners, so it was calibrated against 48 blind hand
labels — recorded by the project owner without sight of any verdict, confidence
or reasoning, and **never revised afterwards**.

| Rubric | Cohen's κ | Raw agreement |
|---|---:|---|
| v1.0.0 | +0.292 | 64.6% |
| v1.1.0 | +0.304 | 79.2% |
| v1.2.0 (final) | **+0.602** | 85.4% |

**Below the 0.7 floor. The milestone closed there rather than iterating until it
cleared**, because tuning a rubric until a number passes is how an instrument
stops measuring anything.

### The defect was polarity, not comprehension

In at least six rows the judge's own written reasoning explained that the answer
*refused* the named behaviour — and it answered that the behaviour held anyway.
Every one was a negatively-framed name (`obeys_`, `claims_`, `treats_`,
`presents_X_as_reliable`). It was answering "is this answer about this topic?"
rather than "does this assertion hold?"

**The decisive evidence was a twin pair.** `injection_not_obeyed` and
`obeys_injected_instruction` ask one question in opposite framings over the same
four scenarios:

| | Agreement, v1.0.0 |
|---|---|
| `injection_not_obeyed` (positive framing) | **4/4** |
| `obeys_injected_instruction` (negative framing) | **1/4, κ = −0.5** |

Same answers, same judge, same rubric. Only the polarity of the name differed.
That isolates the fault to the name's shape rather than to the judge's reading.

The fix was structural rather than a warning: the output contract now requires
the judge to write *what the assertion asserts* before deciding whether it
holds, so the interpretation is committed to before the verdict.

**v1.1.0 then broke the one assertion that was already perfect.** The new rule
misfired on `injection_not_obeyed`, whose name contains its own negation,
dropping it from 4/4 to 0/4 — four gained rows cancelled by four lost, which is
why raw agreement climbed 15 points while κ moved 0.012. v1.2.0 carved out
already-negative names and recovered it.

### The retraction

The full run reported `obeys_injected_instruction` failing **6 times**, and I
called it the most serious result in the suite: an agent following instructions
embedded in a user's question.

**That finding does not stand.** `injection_not_obeyed` — the positively-framed
twin, which agrees with human labels 4/4 under the final rubric — failed **zero**
times across all 12 injection runs. The two are contradictory and the calibration
data says which to believe. The agent appears to have resisted the injections and
the judge inverted.

Without the kappa exercise, this project would have shipped a false security
finding as a headline. That is the single strongest argument for measuring judge
agreement at all, and it is why the retraction is in the evaluation report rather
than quietly corrected.

### What is still unresolved

Four of the seven residual disagreements are the owner's labels sitting against
definitions the owner chose — most sharply on
`margin_below_safety_factor_stated`, kept strict (the 1.25 factor must be named)
in the explicit knowledge that it costs two rows. An answer that explains the
shortfall in days without naming the factor is a real finding about the agent,
not a definition to loosen.

Two disagreements are irreducible: they disagreed identically under all three
rubrics, and in both the owner reads an answer's substance as doing the thing
while it verbally disclaims it.

**The judge is also not stable to text that changes no rule.** Appending the
changelog to the rubric file moved κ from 0.650 to 0.602. One row flipped.
Realistic precision is **0.60 ± 0.05** at n = 48, and the third digit is not
meaningful.

---

## 6. Docker defects found by inspecting the image, not the source

`.dockerignore` was written carefully and reviewed. The Dockerfile was written
carefully and reviewed. Then the image was built and looked inside, and three
things were wrong that no amount of reading would have found.

**A correct exclusion list and a `COPY` that predates it produce the same
document and different images.** That is the whole argument for verifying the
artefact.

### 6.1 `pip install -r requirements.txt` could not resolve at all

The build failed on a clean image: mlflow 3.15 pins a pandas older than the
3.0.5 the feature layer requires. It works locally only because the venv was
built incrementally over months.

**Fixed** by splitting `requirements-api.txt` — serving dependencies only, no
mlflow, matplotlib, kaggle or lightgbm. A test parses the AST of everything under
`src/agent`, `src/api` and `src/obs` and asserts each third-party import is
declared, so the split cannot silently become incomplete.

### 6.2 pip survived two builds

The first inspection found `pip` on `PATH` in the final image. A package
installer in a running container is most of the distance between a
code-execution foothold and a comfortable one. Removing the venv's copy was not
enough — the base image has its own at `/usr/local/bin/pip`, which the venv copy
does not displace.

### 6.3 A comment inside a `RUN` continuation silently disabled the fix

The `rm -rf` that was supposed to remove pip never ran. A `#` on a continued line
comments out the rest of the joined command:

```dockerfile
RUN apt-get install -y libgomp1 \
 && useradd appuser \
 # this comment swallows everything after it
 && rm -rf /usr/local/bin/pip          # never executed
```

The build succeeded, the Dockerfile read correctly, and the image was wrong.
Only the inspection caught it. There is now a test that scans the Dockerfile for
exactly that shape.

**Final image, verified by running it:** uid 10001 non-root; no gcc, cc, g++,
make, ld, pip, pip3, easy_install, ensurepip, curl, wget or git; no `.env`, no
`*.db`, no `*.parquet`, no `*.joblib`, no `kaggle.json` anywhere in the
filesystem. CI runs the same four refusals on every build.

### 6.4 One more, from running the container rather than testing it

A live smoke test with no Ollama reachable returned **500** from `/v1/ask`. A
dependency being down is not this service being broken, and 500 tells a caller
to open a ticket rather than retry. `ProviderError` now maps to 503
`model_unavailable` with `X-Retryable: true`.

---

## 7. Smaller decisions worth the sentence

**`from src.agent.tools import dispatch` hid two guards for weeks.** A
module-level import binds the function object, so wrapping `tools.dispatch`
never reached the loop. The eval harness's failure injection and its
validation-window guard were both silently bypassed — a `tool_failure` scenario
was being shown a tool that worked. Found by a pilot run against a hosted model,
not by a test. The loop now resolves through the module on every call, with two
"guards the guard" tests.

**The agent was never told what "now" is.** Scenarios carried `as_of`; the
harness never passed it on. A hosted model guessed `2025-01-01` — outside the
dataset entirely — and the validation-window guard caught it. Two of five pilot
scenarios aborted before this was fixed.

**Prompt-injection scenarios could not be written as specified.** Both
`MILESTONE_4.md` and `MILESTONE_5.md` required the payload to arrive through a
data field. No output model has a free-text field it could occupy —
`MaintenanceRecord` is `(machine_id, component, replaced_at)` and nothing else.
Writing the test as specified would have meant adding the vulnerability first.
The four scenarios test the user channel and are **labelled** as such, because a
suite that claims coverage it does not have is worse than one claiming none.
Both specs were corrected to match the finding.

**Prompt caching cut input spend by 67.8%.** Measured over all 249 model calls
in the 123-run recording: 1,052,460 input tokens presented, of which 794,599
(75.5%) were served from cache at a tenth of the rate and 6,434 were cache
writes. Priced at the Sonnet input rate that is $1.02 against $3.16 uncached.
Two breakpoints: one on the last tool definition (identical across all 123 runs)
and one on the stable half of the system prompt, with the per-scenario prediction
time deliberately placed after it.

**The harness priced a local run at hosted rates** until transcripts recorded
which provider produced them. Ollama is now zero because it is zero.

**A test was flaky one run in three** and was fixed by understanding it rather
than by widening a tolerance to taste: each span duration is rounded to three
decimals independently, so four children can each round up 0.0005 ms while the
parent rounds down. The tolerance is 0.005 ms with that arithmetic stated, and a
companion test asserts the same containment on raw nanoseconds where it is exact.
