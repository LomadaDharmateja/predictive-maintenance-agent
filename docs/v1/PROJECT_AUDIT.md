> **This document describes the archived v1 of this project (`archive/v1-app/`, `archive/v1-data/`). It is a historical record and does not describe the current codebase.**

# PROJECT_AUDIT.md

Forensic audit of the `industrial-ai-agent` repository (branch `main`, HEAD `cd0149b`).
Audit date: 2026-08-07. Read-only: no source file in the repository was modified.

## Method and its limits

Findings marked as verified were produced by executing code against **copies** of the repository's
data and model artifacts in an isolated scratch environment, never against the working tree.
The repository's own `venv/` is unusable — `venv/pyvenv.cfg` points at
`C:\Users\ldhar\AppData\Local\Programs\Python\Python311`, which no longer exists on this machine —
so a separate Python 3.12.2 environment was built to load the artifacts. Package versions were read
from the `*.dist-info` metadata inside the repository's `venv/Lib/site-packages`, which is the record
of what the project last ran against.

Per the audit constraints, **no external API was called**: Gemini, Pinecone, HuggingFace Inference and
Tavily were not contacted. Anything that depends on those services is marked `UNVERIFIED`.

---

## 1. Repository inventory

### 1.1 Tracked file tree

25 tracked files. Sizes are exact bytes.

```
.devcontainer/devcontainer.json                                       1,054
.github/workflows/pipeline.yml                                        1,471
.gitignore                                                              287
Dockerfile                                                              860
README.md                                                             5,855
requirements.txt                                                      5,738
data/WEG-WMO-Installation-Operation-and-Maintenance-Manual-...pdf  7,347,005
data/commodity.csv                                               18,371,420
data/industrial.csv                                                 329,088
data/industrial_ai.db                                            22,364,160
data/industrial_cleaned.csv                                         240,885
data/maintenance.csv                                                522,048
logs/factory_brain.log                                                4,989
models/failure_predictor.pkl                                      6,886,585
src/app.py                                                            4,652
src/graph_logic.py                                                      953
src/main.py                                                           4,899
src/tools/clean_industrial_data.py                                    1,203
src/tools/data_tools.py                                               5,831
src/tools/db_setup.py                                                 1,548
src/tools/ingest_manual.py                                            1,240
src/tools/search_tools.py                                               758
src/tools/train_model.py                                              1,376
src/tools/vector_tools.py                                             1,338
tests/test_app.py                                                     1,160
```

Present but untracked / ignored: `.env` (ignored, 284 bytes), `venv/` (ignored, 318 installed
distributions), `.pytest_cache/`, `Claude.MD` (untracked; this audit's instruction file).

There is **no `src/__init__.py` and no `src/tools/__init__.py`**. Imports such as
`from tools.data_tools import ...` work only because Streamlit prepends the script's directory to
`sys.path` and Python 3 resolves `tools` as a namespace package. Running `python src/main.py` directly
from the repository root fails.

### 1.2 Source modules

| Module | Lines | Responsibility (one line) |
|---|---:|---|
| `src/app.py` | 112 | Streamlit UI: three tabs, daily-audit fragment, chat loop; owns all session state. |
| `src/main.py` | 117 | `IndustrialAI` class: builds the Gemini LLM, tool list, prompt, agent, memory and executor. |
| `src/graph_logic.py` | 27 | A two-node LangGraph skeleton returning hardcoded strings. Never imported. Dead code. |
| `src/tools/data_tools.py` | 141 | All eight agent tools plus the non-tool `calculate_risk_scores` used by the dashboard. |
| `src/tools/vector_tools.py` | 36 | Builds the HuggingFace-Inference/Pinecone vector store and runs similarity search. |
| `src/tools/search_tools.py` | 21 | Thin Tavily web-search wrapper; instantiates the client at module import. |
| `src/tools/train_model.py` | 40 | Offline script: trains the RandomForest on the SQLite `maintenance` table and pickles it. |
| `src/tools/db_setup.py` | 38 | Offline script: builds `industrial_ai.db` from the three CSVs, inflating maintenance 5×. |
| `src/tools/ingest_manual.py` | 34 | Offline script: loads the WEG PDF, chunks it, pushes embeddings to Pinecone. |
| `src/tools/clean_industrial_data.py` | 29 | Offline script: repairs `industrial.csv` into `industrial_cleaned.csv`. |
| `tests/test_app.py` | 35 | Two tests: Streamlit app starts without exception; `predict_failure` returns a string. |

Total: 630 lines of Python.

### 1.3 Declared dependencies

`requirements.txt` is **UTF-16 LE with a BOM** (first bytes `ff fe`), not UTF-8. It contains 149
requirement lines. Verified: `pip`'s `auto_decode` includes `BOM_UTF16_LE` in its BOM table, so
`pip install -r requirements.txt` still works; but the file is unreadable to most other tooling and to
`open()` without an explicit encoding. It has the shape of a `pip freeze` redirected from PowerShell.

145 of the 149 lines are `==` pins. Four are not: `langchain-pinecone>=0.2.13`, `pinecone>=7.3.0`,
`pinecone-plugin-assistant>=1.8.0`, `pinecone-plugin-interface>=0.0.7`, and `tavily-python` carries no
version constraint at all. That last one has a verified consequence — see §7.3.

**Directly imported by repository code (16):** `streamlit`, `pandas`, `numpy`, `joblib`,
`scikit-learn`, `python-dotenv`, `langchain`, `langchain-classic`, `langchain-core`,
`langchain-google-genai`, `langchain-community`, `langchain-text-splitters`, `langchain-pinecone`,
`langgraph`, `tavily-python`, `pytest`.

Of these, `langgraph` is imported only by `src/graph_logic.py`, which nothing imports.
`scikit-learn` is imported only by the offline `train_model.py`; the serving path reaches sklearn
indirectly through `joblib.load`.

**Declared but not reachable from any imported package (36 of 149)**, computed by walking
`Requires-Dist` metadata from the 16 roots above:

```
aiohttp-retry, asttokens, backcall, beautifulsoup4, bleach, decorator, defusedxml, docopt,
executing, fastjsonschema, ipython, jedi, jupyter_client, jupyter_core, jupyterlab_pygments,
matplotlib-inline, mistune, nbclient, nbconvert, nbformat, pandocfilters, parso, pickleshare,
pipreqs, platformdirs, prompt_toolkit, pure_eval, pyzmq, soupsieve, stack-data, tinycss2,
tornado, traitlets, wcwidth, webencodings, yarg
```

24 of those 36 are the IPython/Jupyter stack. There are no notebooks in the repository.
`pipreqs`/`yarg`/`docopt` are the tool that generated this file and its dependencies.

Additionally, `openai==2.38.0`, `langchain-openai==1.2.2` and `tiktoken==0.13.0` are declared and *are*
reachable (`langchain-openai` pulls the other two), but no code path in the repository uses OpenAI at
all. They are dead weight in the image.

**Required at runtime but NOT declared:** `pypdf`. `src/tools/ingest_manual.py` uses
`PyPDFLoader`, which imports `pypdf` lazily. Verified: `pypdf` appears in neither `requirements.txt`
nor the project's `venv/Lib/site-packages`. The ingestion script therefore cannot run from a clean
install of this repository without an undeclared manual `pip install pypdf`.

---

## 2. Data assets

### 2.1 `data/maintenance.csv` — 522,048 bytes, 10,000 rows × 14 columns

No nulls in any column. No duplicate rows. All 10,000 `Product ID` values are unique — this is a
**one-row-per-unit snapshot, not a time series**, which matters for §5 (`analyze_sensor_trends`
computes a "trend" from a single row).

| Column | dtype | nulls | distinct |
|---|---|---:|---:|
| `UDI` | int64 | 0 | 10000 |
| `Product ID` | str | 0 | 10000 |
| `Type` | str | 0 | 3 |
| `Air temperature [K]` | float64 | 0 | 93 |
| `Process temperature [K]` | float64 | 0 | 82 |
| `Rotational speed [rpm]` | int64 | 0 | 941 |
| `Torque [Nm]` | float64 | 0 | 577 |
| `Tool wear [min]` | int64 | 0 | 246 |
| `Machine failure` | int64 | 0 | 2 |
| `TWF` | int64 | 0 | 2 |
| `HDF` | int64 | 0 | 2 |
| `PWF` | int64 | 0 | 2 |
| `OSF` | int64 | 0 | 2 |
| `RNF` | int64 | 0 | 2 |

**Class balance of the classification target `Machine failure`:**

| value | count | share |
|---|---:|---:|
| 0 | 9,661 | 96.61 % |
| 1 | 339 | 3.39 % |

A majority-class classifier scores **96.61 % accuracy**. Any accuracy figure below that is worse than
predicting "no failure" every time. Per-mode counts: `TWF` 46, `HDF` 115, `PWF` 95, `OSF` 98, `RNF` 19.
`Type` splits L 6,000 / M 2,997 / H 1,003.

Label inconsistencies, all verified by counting:
- **9 rows** have `Machine failure == 1` but no failure-mode flag set.
- **18 rows** have a flag set but `Machine failure == 0`; 18 of the 19 `RNF == 1` rows are in this group.
- **24 rows** have more than one flag set simultaneously.

Feature ranges: air temp 295.3–304.5 K (μ 300.00), process temp 305.7–313.8 K (μ 310.01),
speed 1168–2886 rpm (μ 1538.8), torque 3.8–76.6 Nm (μ 39.99), tool wear 0–253 min (μ 107.95).

`UNVERIFIED:` This is almost certainly the **AI4I 2020 Predictive Maintenance Dataset** (UCI ML
Repository, synthetic, CC BY 4.0). The column names, the exact 10,000/339 split, the L/M/H product
quality mix and the TWF/HDF/PWF/OSF/RNF mode encoding all match its published description. No source,
citation or licence is recorded anywhere in the repository, so this is an identification by
fingerprint, not a documented provenance. `src/tools/data_tools.py:40` calls them "the standard column
names for the AI4I dataset", which is the only mention in the code and is itself just a comment.

**No ingestion or generation script for this file exists.** It is a committed artifact
(file mtime 2022-11-06, i.e. older than the repository).

### 2.2 `data/commodity.csv` — 18,371,420 bytes, 49,093 rows × 29 columns

Date range 1960-01-01 to 2026-02-01, 71 distinct commodities, 242–794 rows per commodity.
`data_source` is self-documenting: 49,013 rows `World Bank Pink Sheet`, 80 rows
`FRED (Federal Reserve St. Louis)`. Units: `($/mt)` 22,642, `($/kg)` 14,542, `($/bbl)` 2,884,
`($/cubic meter)` 2,628, `($/troy oz)` 2,340, `($/mmbtu)` 2,149, `($/dmtu)` 780, `(2010=100)` 576,
`(cents/sheet)` 552.

Null counts (only non-zero columns shown): `source_desc` 4,882; `dataset_version` 80;
`retrieved_date` 80; `build_timestamp` 80; `price_mom_pct` 71; `price_yoy_pct` 852; `price_mom_abs` 71;
`price_12m_avg` 142; `price_60m_avg` 781; `price_12m_volatility` 355; `price_index_2000_base` 787;
`commodity_code` 24,156. All other columns are complete. There is no classification target in this file.

Two data-integrity problems, verified:
- **The `category` column is wrong for metals.** `Aluminum`, `Nickel`, `Zinc`, `Tin`, `Lead`,
  `Iron ore, cfr spot` and `Urea` are all labelled `Fertilizers`. `Copper` appears under two
  categories (`Fertilizers` and `Metals & Minerals`) and is the only commodity with more than one.
  The whole `Metals & Minerals` category has 13 rows.
- **The file is not sorted by date.** It is grouped by commodity, then by date. The maximum date in
  the file (2026-02-01) belongs to `Crude oil, Brent` / `Crude oil, WTI`, but the file's *last* rows
  are Zinc through 2024-12-01. This breaks the dashboard chart (§8) and makes `.iloc[-1]` in
  `get_internal_commodity_prices` a "last row of this commodity's block", not a "latest price".

`UNVERIFIED:` The World Bank Pink Sheet is the stated origin inside the data, but no download or
build script is committed and the `build_timestamp` (2026-03-16 16:15:07) refers to a build process
that is not in this repository. The `dataset_version`, `retrieved_date`, `build_timestamp`,
`row_completeness_pct` (constant 100.0) and `era`/`decade` columns indicate this CSV was produced by a
separate pipeline that was never checked in.

### 2.3 `data/industrial.csv` — 329,088 bytes, 2,342 rows × 22 columns

Supply-chain table: supplier, manufacturer, logistics, retailer fields plus a three-class
`Optimization_Label` (`Low` 1,248 / `Medium` 879 / `High` 215). 50 distinct `Supplier_ID`. No nulls.

**The `Actual_Demand` column is corrupt in a specific and revealing way.** Every one of the 2,342
values is the identical string:

```
<function <lambda> at 0x0000020C0EA29DA0>
```

That is a Python function `repr` — a data-generation script wrote a lambda object into a DataFrame
column instead of calling it, and the memory address proves the whole column was written in one pass
from one process. This is direct evidence the file was **synthesised, not collected**.

`UNVERIFIED:` best guess is a Kaggle-style synthetic supply-chain optimisation dataset. Nothing in
the repository names it. The generation script is not committed.

### 2.4 `data/industrial_cleaned.csv` — 240,885 bytes, 2,342 rows × 22 columns

Produced by `src/tools/clean_industrial_data.py` — the one data script that *is* committed. Verified:
every column except `Actual_Demand` is byte-identical to `industrial.csv`. `Actual_Demand` is now
int64, and equals `Forecasted_Demand × Uniform(0.9, 1.1)` truncated to int — measured ratio range
[0.8996, 1.0998], mean 0.9993, correlation with `Forecasted_Demand` 0.851.

So the "actual demand" that the agent reasons about is **a random jitter of the forecast, generated at
cleaning time with no seed**. Re-running the cleaner produces different numbers. The script's own
comment says "to make it look like real-world fluctuation" (`clean_industrial_data.py:12-13`).

### 2.5 `data/industrial_ai.db` — 22,364,160 bytes, SQLite

Built by `src/tools/db_setup.py`. Three tables:

| Table | Rows | Cols | Source |
|---|---:|---:|---|
| `maintenance` | 50,000 | 14 | `maintenance.csv` concatenated 5× |
| `commodities` | 49,093 | 29 | `commodity.csv` verbatim |
| `logistics` | 2,342 | 22 | `industrial_cleaned.csv` verbatim |

Verified: every `UDI` appears exactly 5 times (value counts are uniformly 5). `Machine failure`
balance is 48,305 / 1,695 — the same 3.39 % rate, five copies of it. `db_setup.py:20-23` calls this
"Inflating dataset to 50,000 rows for robustness" and adds `N(0, 2)` noise to `Torque [Nm]` only, so
the 5 copies are not exact duplicates (0 exact duplicate rows) but are near-duplicates differing in one
column. No random seed is set, so the DB is not reproducible.

This inflated table is what the shipped model was trained on. Consequences in §3.

No index, primary key or constraint is created on any table.

### 2.6 `data/WEG-WMO-Installation-Operation-and-Maintenance-Manual-of-Electric-Motors.pdf`

7,347,005 bytes, **54 pages**, 164,386 extractable characters. PDF metadata: Creator
`Adobe InDesign 18.1 (Windows)`, Producer `Adobe PDF Library 17.0`, CreationDate 2023-05-31.
First page text confirms: "Installation, operation and maintenance manual of electric motors …
This manual provides information about WEG induction motors fitted with squirrel cage, permanent
magnet or hybrid rotors".

This is a vendor manual published by WEG. No licence statement accompanies it in the repository, and
no permission to redistribute is documented. This is **the entire knowledge base** — the RAG layer has
exactly one source document.

### 2.7 `models/failure_predictor.pkl` — 6,886,585 bytes

See §3.

### 2.8 `logs/factory_brain.log` — 4,989 bytes

A committed runtime log covering 2026-04-14 to 2026-05-26. It contains no secrets, but it does
contain evidence used elsewhere in this audit: real `POST` requests to
`generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent` (April) and to
`gemini-3.1-flash-lite:streamGenerateContent` (May), plus a 503 retry sequence. Committing a runtime
log is a hygiene problem: `logs/` is not in `.gitignore`, so every run dirties the working tree.

---

## 3. Machine learning component

### 3.1 The serialised model — loaded and inspected

`models/failure_predictor.pkl`, 6,886,585 bytes, written with `joblib.dump` (uncompressed pickle
protocol 4; the stream begins `\x80\x04\x95` and contains `joblib.numpy_pickle.NumpyArrayWrapper`).

- **Exact class:** `sklearn.ensemble._forest.RandomForestClassifier`
- **Pickled with scikit-learn 1.7.2.** Verified twice: the string `1.7.2` follows `_sklearn_version`
  in the raw pickle stream, and loading under scikit-learn 1.8.0 raises
  `InconsistentVersionWarning: Trying to unpickle estimator RandomForestClassifier from version 1.7.2
  when using version 1.8.0`. Loading under 1.7.2 raises no version warning.
- **`requirements.txt` pins `scikit-learn==1.8.0`.** So a clean install of this repository loads the
  model with a version-mismatch warning on every call. Verified that predictions are unchanged
  between 1.7.2 and 1.8.0 for the probes tried, but sklearn's own documentation makes no such
  guarantee across minor versions.

**Hyperparameters** (complete `get_params()`):

```
bootstrap = True                   min_samples_leaf = 1
ccp_alpha = 0.0                    min_samples_split = 2
class_weight = None                min_weight_fraction_leaf = 0.0
criterion = 'gini'                 monotonic_cst = None
max_depth = None                   n_estimators = 100
max_features = 'sqrt'              n_jobs = None
max_leaf_nodes = None              oob_score = False
max_samples = None                 random_state = 42
min_impurity_decrease = 0.0        verbose = 0
                                   warm_start = False
```

`class_weight = None` on a 3.4 %-positive target: the model is not rebalanced in any way.
`max_depth = None` with `min_samples_leaf = 1`: trees grow until pure. Measured across the 100
estimators — depth 19 to 27 (mean 22.44), leaves 363 to 530 (mean 428.4).

**Features — exact names and order**, read from `feature_names_in_`:

```
0  Air temperature [K]
1  Process temperature [K]
2  Rotational speed [rpm]
3  Torque [Nm]
4  Tool wear [min]
```

`n_features_in_ = 5`, `classes_ = [0 1]`, `n_classes_ = 2`, `n_outputs_ = 1`, 100 fitted estimators.
Gini importances: Torque 0.2458, Rotational speed 0.2400, Air temperature 0.1827, Tool wear 0.1706,
Process temperature 0.1610.

### 3.2 Training script

`src/tools/train_model.py` exists (40 lines) and is a plausible producer of this artifact: it reads
`SELECT * FROM maintenance` from `industrial_ai.db`, selects exactly those five columns in exactly
that order, targets `Machine failure`, and fits `RandomForestClassifier(n_estimators=100,
random_state=42)`. The recovered hyperparameters match.

So the model is nominally reproducible — but only nominally:

1. `db_setup.py` adds `np.random.normal(0, 2, ...)` to `Torque [Nm]` **with no seed**. The training
   table cannot be reconstructed. Re-running `db_setup.py` then `train_model.py` produces a
   *different* model.
2. The training script trains on the **5×-inflated 50,000-row table**, not the 10,000-row source.
3. There is no `if __name__` guard problem, no CLI, no config, and no record of which sklearn version
   or DB build produced the committed `.pkl`. The `.pkl` is dated 2022-11-06… no — file mtime is
   2026-04-08 22:53, identical to `industrial_ai.db`'s mtime, which is consistent with a single
   `db_setup.py` → `train_model.py` run that day.

**There is no evaluation code anywhere in the repository.** No train/test split, no cross-validation,
no held-out set, no metric computation, no confusion matrix, no threshold selection. `train_model.py`
calls `model.fit(X, y)` on 100 % of the data and immediately `joblib.dump`s it. No metric for this
model is recorded anywhere in the repository, including the README.

Measurements made by this audit (these are **not** in the repository):

- Resubstitution on the exact 50,000-row training table: accuracy 1.0, precision 1.0, recall 1.0,
  confusion matrix `[[48305, 0], [0, 1695]]`. This is what an unpruned forest does on its own training
  data and says nothing about generalisation.
- Honest 5-fold stratified CV, same hyperparameters, on the **original 10,000 rows**:
  precision 0.8555, recall 0.6460, F1 0.7361, ROC-AUC 0.9667, confusion matrix
  `[[9624, 37], [120, 219]]`. It misses 120 of 339 real failures.
- The same procedure on the **inflated 50,000-row table** with a naive random 80/20 split:
  F1 0.9639, recall 0.9440. The gap between 0.736 and 0.964 is the 5× duplication leaking
  near-identical rows across the split. Anyone who evaluated this model the obvious way — random split
  on the DB table the training script reads — would have measured a number roughly 23 F1 points too
  optimistic.

### 3.3 How the model is invoked at inference

`src/tools/data_tools.py:13-27`, tool `predict_failure`:

```python
model = joblib.load(MODEL_PATH)
features = [[air_temp, process_temp, speed, torque, tool_wear]]
probability = model.predict_proba(features)[0][1]
```

- The 6.9 MB pickle is **re-loaded from disk on every single call**. No caching, no module-level load.
- The input is a bare nested list, so the feature names recorded in `feature_names_in_` are discarded.
  Verified: this raises `UserWarning: X does not have valid feature names, but RandomForestClassifier
  was fitted with feature names` on every call.
- **Feature ordering is therefore positional and unchecked.** It happens to match training order today
  because the parameter order of `predict_failure` mirrors the column order in `train_model.py:25`.
  Nothing enforces that. Verified the cost of getting it wrong: for the row
  `(298.1, 308.6, 1551 rpm, 42.8 Nm, 0 min)` the correct-order probability is **0.00**; swapping speed
  and torque yields **0.70** — the difference between "STABLE" and "WARNING". A one-line reordering of
  the signature would silently invert the system's conclusions.
- There is no input validation. Verified: `air_temp=-999, process_temp=0, speed=-5, torque=1e9,
  tool_wear=-3` returns `Failure Probability: 72.00%. Risk Status: WARNING.` — physically impossible
  inputs produce a confident-looking answer.
- The risk bands are hardcoded at `data_tools.py:26`: `> 0.8` CRITICAL, `> 0.5` WARNING, else STABLE.
  These thresholds were not derived from any measurement; with 3.4 % base rate and no calibration,
  a 0.5 threshold on an uncalibrated forest is arbitrary.

`predict_failure` is registered as an agent tool but is **not called anywhere in the UI**. The
"Predictive" tab uses `calculate_risk_scores`, which is a hand-written rule, not the model (§5.9).
The model only ever runs if the LLM decides to call it.

---

## 4. Agent layer

### 4.1 Model and provider

`src/main.py:57-61`:

```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)
```

- **Provider:** Google Gemini Developer API, via `langchain-google-genai==4.2.3`.
- **Model identifier:** `gemini-3.1-flash-lite`.
- `logs/factory_brain.log` confirms live traffic to
  `.../models/gemini-3.1-flash-lite:streamGenerateContent` on 2026-05-25 and 2026-05-26, so this
  identifier resolved successfully at that time.
- **The UI and README disagree with the code.** `src/app.py:37` renders
  `**Brain:** Gemini 2.5 Flash-Lite` in the sidebar, and the README says "Gemini 2.5 Flash-Lite" twice.
  Both are stale; the code has said `3.1-flash-lite` since commit `7695407`.

### 4.2 Framework and construction pattern

- `langchain-core==1.4.0`, `langchain==1.3.1`, `langchain-classic==1.0.7`.
- Pattern: **`create_tool_calling_agent` + `AgentExecutor`** (`main.py:87, 91`), the legacy
  `langchain_classic` agent API — not `langgraph`'s `create_agent`, despite `langgraph==1.2.1` being
  installed and `src/graph_logic.py` existing.
- `AgentExecutor` is subclassed as `CleanAgentExecutor` (`main.py:44-50`) purely to post-process
  `response["output"]` through `clean_gemini_output`, which unwraps Gemini's list-of-content-blocks
  form into plain text.

### 4.3 System prompt, verbatim

From `src/main.py:76-81`, reproduced exactly including its literal indentation:

```
You are a Senior Industrial Systems Engineer at VULCAN OS. 
            You have autonomous access to sensors, manuals, and market data.
            Rules:
            1. If a sensor is abnormal, check the technical manual using 'consult_technical_manual'.
            2. If a part needs replacement, check 'get_internal_commodity_prices' or 'get_supplier_info'.
            3. Always provide a factual, engineering-based reasoning.
```

The prompt template is:

```python
ChatPromptTemplate.from_messages([
    ("system", <the above>),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

Observations: the prompt names only 3 of the 9 registered tools. It gives no output format, no
instruction about citing sources, no instruction on what to do when a tool returns an error string,
and no statement that the data is synthetic. The trailing whitespace and 12-space indentation on
lines 2-6 are sent to the model as-is.

### 4.4 Registered tools

Nine tools, all from `src/tools/data_tools.py`, registered at `main.py:63-73`. Signatures and
docstrings are reproduced exactly as written in the source:

| # | Signature | Docstring (verbatim) |
|---|---|---|
| 1 | `analyze_sensor_trends(product_id: str, sensor_name: str)` | `Checks if a machine's current sensor reading deviates from the factory average.` |
| 2 | `check_maintenance_sensors(product_id: str)` | `Retrieves real-time sensor data (temp, speed, torque) for a specific machine ID.` |
| 3 | `consult_technical_manual(query: str)` | `Searches the WEG Motor Manual (Pinecone) for repair steps and technical guides.` |
| 4 | `run_sql_query(query: str)` | `Execute SQL queries against the 'industrial_ai.db'. Use for complex cross-table analysis.` |
| 5 | `get_market_news(query: str)` | `Fetches real-time supply chain disruptions or price forecasts using Tavily.` |
| 6 | `predict_failure(air_temp: float, process_temp: float, speed: float, torque: float, tool_wear: float)` | `Predicts the probability of a machine failure based on sensor inputs.`<br>`Inputs: Air Temp (K), Process Temp (K), Speed (rpm), Torque (Nm), Tool Wear (min).` |
| 7 | `get_failed_machines(query: str = "")` | `Returns a list of failed machines from maintenance.csv with decoded failure types.` |
| 8 | `get_supplier_info(query: str = "")` | `Checks the internal logistics database for reliable suppliers. Input is optional.` |
| 9 | `get_internal_commodity_prices(commodity_name: str)` | `Checks the LOCAL commodity.csv for the last recorded price of a material.` |

None of the docstrings document return format, failure behaviour, units of the return value, or the
fact that the data is a static snapshot. Two docstrings are actively misleading: `check_maintenance_sensors`
promises "real-time sensor data" from a static CSV of 2022 vintage, and `get_market_news` promises
"real-time" data that depends on an external API that may return an error string.

`get_failed_machines` and `get_supplier_info` take a `query` parameter that is accepted and then never
used — the function body ignores it entirely.

### 4.5 Memory

`main.py:88`: `ConversationBufferMemory(memory_key="chat_history", return_messages=True)`.

- **Unbounded.** `ConversationBufferMemory` keeps every message of the session verbatim. There is no
  window, no summary, no token cap, no trimming. Every turn resends the entire history, including the
  full text of every prior tool result. Given that `run_sql_query` can return an 8.5 MB string (§5.4)
  and `get_failed_machines` returns a 16,647-character string, the context grows fast and
  monotonically until the Gemini request fails.
- Verified deprecated: constructing it emits `LangChainDeprecationWarning: The class
  ConversationBufferMemory was deprecated in LangChain 0.3.1 and will be removed in 2.0.0.`
- The memory lives on the `IndustrialAI` instance in `st.session_state.agent`, so it persists for the
  lifetime of the browser session and is never cleared. There is no "reset conversation" control.
- History is stored **twice**: once in `ConversationBufferMemory` and again in
  `st.session_state.messages` (`app.py:88`). The two can diverge (§8).

### 4.6 Generation and iteration parameters

- `temperature = 0.2`. Set explicitly; changed in the most recent commit (`cd0149b "changed temperature"`).
- No `max_tokens`, `top_p`, `top_k`, `timeout`, `max_retries`, `safety_settings` or `stop` are set —
  all library defaults.
- `AgentExecutor` is constructed with only `agent`, `tools`, `memory`, `verbose=True`. Therefore
  the class defaults apply, read from `langchain_classic/agents/agent.py:1023-1032`:
  **`max_iterations = 15`**, **`max_execution_time = None`** (no wall-clock limit),
  `early_stopping_method = "force"`, `handle_parsing_errors = False`, `return_intermediate_steps = False`.
- `verbose=True` in a Streamlit app writes the full agent trace to stdout only; the user never sees it,
  and it is not routed to the `logs/factory_brain.log` logger.
- No `handle_tool_error` is configured on any tool, so a tool that raises aborts the whole run (§5).

### 4.7 Are tool outputs validated?

**No.** There is no validation, schema check, size limit, sanitisation or truncation anywhere between
a tool's `return` and the model's context. Verified specifics:

- Return types are inconsistent: seven tools return `str`, `check_maintenance_sensors` returns a
  `dict`. LangChain's `_format_output` passes non-string content through `_stringify`
  (`json.dumps`, falling back to `str`), so the dict is silently JSON-encoded. It works, but by
  accident rather than design.
- Error conditions are returned as **ordinary success strings**. `search_manual` returns
  `f"Error searching technical manual: {e}"`, and `run_sql_query` returns
  `f"Error in SQL query: {e}"`. To the model these are indistinguishable from real content. The
  system prompt gives no guidance for that case, so the model is free to proceed as though the manual
  had been consulted.
- No output is size-capped. See §5.4.
- `check_maintenance_sensors` returns **all 14 columns**, including `Machine failure`, `TWF`, `HDF`,
  `PWF`, `OSF`, `RNF`. The ground-truth labels are handed to the model as "sensor data". Any
  "prediction" the agent makes about a unit it has looked up is reading the answer key.

---

## 5. Tool implementations

### 5.1 `predict_failure` — `data_tools.py:13-27`

**Does:** loads the pickled RandomForest and returns a failure probability plus a band label.
**Touches:** `models/failure_predictor.pkl` (absolute path derived from `__file__`, so cwd-independent).
**Returns:** `f"Failure Probability: {probability:.2%}. Risk Status: {risk_level}."` — e.g.
`Failure Probability: 0.00%. Risk Status: STABLE.`
**Validation:** only `os.path.exists(MODEL_PATH)`. No range, sign or type checks on the five inputs;
Pydantic coerces numeric strings, so `"298"` is accepted.
**On failure:** if the file is missing it returns a polite string. If the file exists but is corrupt or
the sklearn version is incompatible, `joblib.load` raises and the exception propagates out of the tool,
out of `AgentExecutor`, and into Streamlit as an unhandled traceback.

### 5.2 `get_failed_machines` — `data_tools.py:31-56`

**Does:** filters `Machine failure == 1` and assigns each row the **first** flag it finds among
`['TWF','HDF','PWF','OSF','RNF']`.
**Touches:** `data/maintenance.csv` via a **relative path** — breaks if cwd is not the repo root.
**Returns:** `str()` of a list of 339 dicts, 16,647 characters, e.g.
`[{'product_id': 'L47230', 'failure_type': 'PWF'}, ...]`.
**Validation:** none. The `query` parameter is accepted and ignored.
**On failure:** `FileNotFoundError` propagates unhandled.

Verified defect: because 24 rows carry multiple flags and the loop `break`s on the first hit, the
returned distribution does not match the data. Tool output: HDF 115, PWF 91, OSF 78, TWF 46,
**Unknown 9**, RNF 0. Actual column sums: HDF 115, PWF 95, OSF 98, TWF 46, RNF 19. The tool
under-reports OSF by 20 and never reports RNF at all. The 9 `Unknown` entries are the label-inconsistent
rows from §2.1 — the code emits the literal string `"Unknown"` as a failure type with no explanation.

### 5.3 `get_internal_commodity_prices` — `data_tools.py:58-66`

**Does:** substring-matches `commodity_name` against the `commodity_name` column and reports the last
matching row.
**Touches:** `data/commodity.csv` (relative path), re-read in full — 18.4 MB parsed per call.
**Returns:** `f"Internal Record: {commodity_name} is {latest['price_nominal_usd']} per {latest['unit']}."`
or `"Material not found in local records."`
**Validation:** none.

Three verified defects:
- **The returned string echoes the caller's search term, not the matched commodity.** Query `"oil"`
  returns `Internal Record: oil is 1222.5 per ($/mt).` — the price of whichever oil sorted last in the
  file, presented as the price of "oil". The model has no way to know which commodity it received.
- **`str.contains` defaults to `regex=True`.** A query containing a regex metacharacter raises. Verified:
  `get_internal_commodity_prices("(")` raises `re.error: missing ), unterminated subpattern at position 0`,
  which propagates out of the agent run. This is user-controlled input reaching a regex compiler.
- **`.iloc[-1]` is not "latest".** The file is ordered by commodity then date (§2.2), so this is the
  last row of that commodity's block. For Copper it happens to be 2026-01-01 (12986.6068 $/mt), which
  is correct by luck of the file layout, not by design. No `sort_values('date')` is performed.

Also: `"Steel"` returns `Material not found in local records.` The README claims the system tracks
"material costs (Copper/Steel)". Verified: **there is no steel series in `commodity.csv`** — 0 matching
rows across all 71 commodities.

### 5.4 `run_sql_query` — `data_tools.py:126-136`

**Does:** executes arbitrary caller-supplied SQL against the SQLite file.

```python
conn = sqlite3.connect('data/industrial_ai.db')
try:
    result = pd.read_sql_query(query, conn)
    return result.to_string()
except Exception as e:
    return f"Error in SQL query: {e}"
finally:
    conn.close()
```

**Touches:** `data/industrial_ai.db` (relative path).
**Returns:** `DataFrame.to_string()` — a whitespace-aligned text table.
**Validation:** **none whatsoever.** The SQL string comes from the LLM, which in turn is driven by
untrusted user text from the chat box. There is no allowlist, no statement-type check, no read-only
connection (`sqlite3.connect('file:...?mode=ro', uri=True)` is not used), no `LIMIT` injection and no
row cap.
**On failure:** returns the exception text as a normal string, which the model cannot distinguish from
a result.

Two verified consequences, both measured against a **copy** of the database:

1. **Destructive DDL succeeds while the tool reports an error.** `DROP TABLE logistics` returned
   `Error in SQL query: 'NoneType' object is not iterable` — pandas failing on a `None`
   `cursor.description` — but the table was gone afterwards
   (`SELECT name FROM sqlite_master` → `['maintenance', 'commodities']`). The `except` swallows the
   pandas error, the `finally` closes the connection, and SQLite has already auto-committed the DDL.
   By contrast `DELETE FROM logistics` left all 2,342 rows intact, because DML opens an implicit
   transaction that is rolled back on close. So the tool is destructive specifically for the
   statements that do the most damage — `DROP`, and by the same mechanism `ALTER`.
2. **Unbounded output straight into the LLM context.** `SELECT * FROM maintenance` returns a string of
   **8,550,170 characters**. That is fed verbatim to Gemini as a tool result and appended to the
   unbounded `ConversationBufferMemory`. One such call ends the session.

### 5.5 `get_supplier_info` — `data_tools.py:68-73`

**Does:** filters `Reliability_Score > 0.8`, sorts by `Lead_Time_Supplier`, returns the first three.
**Touches:** `data/industrial_cleaned.csv` (relative path). Note the docstring says "internal logistics
database" but it reads the CSV, not the `logistics` table in SQLite.
**Returns:** `str()` of a column-oriented dict keyed by the original DataFrame index, e.g.
`{'Supplier_ID': {2324: 'S026', 2287: 'S025', 2293: 'S015'}, 'Lead_Time_Supplier': {2324: 2, ...}, 'Reliability_Score': {...}}`
— a format that requires the model to join three dicts by integer index to recover one supplier row.
**Validation:** none. `query` is accepted and ignored.
**On failure:** `FileNotFoundError`/`KeyError` propagate unhandled.

Verified: 1,490 of 2,342 rows pass the `> 0.8` filter, and rows are not deduplicated by `Supplier_ID`
(only 50 distinct suppliers exist), so the "top 3 suppliers" is really "3 arbitrary rows among 1,490
tied at lead time 2". The thresholds 0.8 and 3 are hardcoded.

### 5.6 `check_maintenance_sensors` — `data_tools.py:98-103`

**Does:** exact-matches `Product ID` and returns the whole row.
**Touches:** `data/maintenance.csv` (relative path).
**Returns:** a **`dict`** (not a string) from `DataFrame.to_dict()`, or the string
`"Machine ID not found."` LangChain JSON-encodes the dict on the way to the model.
**Validation:** none; `product_id` is used for an exact equality match, so no injection surface, but
also no normalisation (case, whitespace).
**On failure:** file errors propagate unhandled.

Verified: the returned dict contains all 14 columns including the five failure-mode labels and
`Machine failure` (§4.7). Because `Product ID` is unique across all 10,000 rows, "real-time sensor
data" is a single static row.

### 5.7 `consult_technical_manual` — `data_tools.py:120-124` → `vector_tools.py:28-36`

**Does:** delegates to `search_manual`, which builds a `PineconeVectorStore` and runs
`similarity_search(query, k=3)`, joining the three chunks with newlines.
**Touches:** HuggingFace Inference API (embedding the query) and Pinecone index `vulcan-manuals`.
**Returns:** the concatenated `page_content` of 3 chunks, or the string
`f"Error searching technical manual: {str(e)}"`.
**Validation:** none on the query. `get_vectorstore` does check `HUGGINGFACE_API_KEY` and raises a
`ValueError` if absent — but `search_manual` catches everything, so a missing key becomes an error
string handed to the model as though it were manual content.
**On failure:** **silent degradation.** Network error, expired key, deleted index, HF model
unavailability — all collapse to one string that flows into the LLM's context with no signal that
retrieval failed. This is the single most dangerous failure mode in the system: the agent will keep
answering technical repair questions with the retrieval step silently broken.

The returned chunks carry **no page numbers or source metadata**, even though the loader populates
`page` and `source` in each chunk's metadata. Citations are therefore impossible; the README's claim
of "grounded technical instructions" cannot be checked by a user.

`UNVERIFIED:` behaviour against the live index — no Pinecone or HuggingFace call was made.

### 5.8 `get_market_news` — `data_tools.py:138-142` → `search_tools.py:10-21`

**Does:** `tavily.search(query=query, search_depth="advanced", max_results=5)`, then concatenates
`Source: {url}\nContent: {content}` for each result.
**Touches:** the Tavily API.
**Returns:** a concatenated string of up to 5 results, unbounded in length.
**Validation:** none, in either direction. Two `print()` calls write to stdout, not the logger.
**On failure:** **no try/except at all.** Any Tavily error — auth, rate limit, timeout, network,
or a response without a `results` key — raises `KeyError`/`requests` exceptions that propagate out of
the tool, out of `AgentExecutor`, and surface in Streamlit as a traceback. Unlike the manual tool,
this one fails loudly and kills the whole turn.

There is no timeout override (the client default is 60 s) and no retry.

### 5.9 `calculate_risk_scores` — `data_tools.py:75-95` (not a registered tool)

**Does:** computes a hand-written "health score" for the dashboard. Not exposed to the agent.
**Touches:** `data/maintenance.csv`.
**Returns:** a DataFrame of the **last 10 rows** with `Product ID`, `health_score` and three risk
components.

```python
temp_diff = df['Process temperature [K]'] - df['Air temperature [K]']
df['heat_risk']  = np.where((temp_diff < 8.6) & (df['Rotational speed [rpm]'] < 1380), 1.0, 0.2)
power = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * (2 * np.pi / 60))
df['power_risk'] = np.where((power < 3500) | (power > 9000), 1.0, 0.1)
overstrain = df['Tool wear [min]'] * df['Torque [Nm]']
df['overstrain_risk'] = overstrain / 11000
df['health_score'] = (1 - df[['heat_risk','power_risk','overstrain_risk']].max(axis=1)) * 100
```

The thresholds 8.6 K, 1380 rpm, 3500/9000 W and 11000 Nm·min are the published generation rules of the
AI4I dataset, hardcoded with no citation. Note this means the dashboard's "health" is **not** the ML
model's opinion; the model and the dashboard are two unrelated systems that never meet.

**Verified defect with a visible consequence:** for the last 10 rows of `maintenance.csv`, all three
risk components come out at heat 0.2, power 0.1, overstrain ≤ 0.11, so `max` is always 0.2 and
`health_score` is **exactly 80.0 for every one of the 10 units** (std 0.0). `app.py:46` gates the
critical alert on `top_issue['health_score'] < 80` — strictly less than. 80.0 is not < 80. Therefore
**the "CRITICAL" branch, the health metric and the "Generate Repair Strategy" button are unreachable
for this dataset**; the Daily Autonomous Audit always renders "All units operating within normal
parameters", and the Predictive tab always shows five units at 80 %.

---

## 6. Retrieval layer

| Property | Value | Evidence |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `vector_tools.py:17`, `ingest_manual.py:23` |
| Embedding access | `HuggingFaceInferenceAPIEmbeddings` (remote HTTP), default endpoint `https://api-inference.huggingface.co/pipeline/feature-extraction/{model}` | class source in `langchain_community/embeddings/huggingface.py` |
| Dimensionality | 384 | `UNVERIFIED` — the documented output width of all-MiniLM-L6-v2; not asserted anywhere in the repo and not confirmed against the live index |
| Vector store | Pinecone, index name `vulcan-manuals` | `vector_tools.py:21-24` |
| Index config (metric, dims, pods/serverless, region, namespace) | not in the repository | `UNVERIFIED` — no Pinecone call made |
| Retrieval | `similarity_search(query, k=3)`, default metric, no filters, no score threshold, no MMR | `vector_tools.py:33` |
| Chunking | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)`, default separators | `ingest_manual.py:17` |
| Source documents | Exactly one: the 54-page WEG motor manual | `ingest_manual.py:12` |

**Chunk count.** Verified locally by running the identical loader and splitter against the committed
PDF, with no network access: `PyPDFLoader` yields **54 documents** (one per page) and the splitter
produces **204 chunks**, min 48 / mean 842.7 / max 1000 characters. This is what the ingestion script
*would* produce today. Whether the live `vulcan-manuals` index actually contains 204 vectors is
`UNVERIFIED`.

**Reproducibility of the index — three separate blockers:**

1. `pypdf` is undeclared (§1.3), so `ingest_manual.py` cannot run from a clean install.
2. `langchain_pinecone` does **not create indexes**. Verified in
   `langchain_pinecone/vectorstores.py:912-925`: if the named index is absent it raises
   `Index '<name>' not found in your Pinecone project. Did you mean one of the following indexes: ...`.
   No code in this repository creates `vulcan-manuals`, sets its dimension, or sets its metric. **The
   index must have been created by hand in the Pinecone console**, and its configuration exists only
   in that account.
3. `ingest_manual.py` is not idempotent. `PineconeVectorStore.from_documents` generates fresh UUIDs
   per chunk, so running it twice duplicates all 204 chunks. There is no delete-before-ingest, no
   namespace, and no ID derivation from content.

Additional risks:
- `vector_tools.py` never calls `load_dotenv()`. It works only because `main.py` (or `search_tools.py`)
  imported first and loaded the `.env`. Importing `vector_tools` on its own picks up nothing from `.env`.
- `langchain-community` is sunset. Verified: importing it emits
  `DeprecationWarning: langchain-community is being sunset and is no longer actively maintained`.
  `HuggingFaceInferenceAPIEmbeddings` lives there; its sibling `HuggingFaceEmbeddings` in the same
  module is already marked `@deprecated(since="0.2.2", removal="1.0")`.
- `UNVERIFIED:` whether the legacy `api-inference.huggingface.co` feature-extraction endpoint still
  serves this model. HuggingFace has been migrating serverless inference to `router.huggingface.co`
  and retiring models from the free tier. If that endpoint has moved, retrieval fails into the silent
  error-string path of §5.7.
- Query and document embeddings must come from the same model for the index to be meaningful.
  `ingest_manual.py:23` and `vector_tools.py:17` do use the same model name, and the comment
  `# Match retrieval` shows this was a deliberate fix. That part is correct.

---

## 7. External services and configuration

### 7.1 External APIs

| Service | Called from | Purpose | Trigger |
|---|---|---|---|
| Google Gemini Developer API | `main.py:57` via `langchain-google-genai` | Every agent turn | Chat message, "Run Executive System Audit", "Generate Repair Strategy" |
| Tavily Search API | `search_tools.py:14` | Web search | `get_market_news` tool, and the sidebar "Enable Live Supply Chain Feed" checkbox |
| HuggingFace Inference API | `vector_tools.py:14`, `ingest_manual.py:21` | Embeddings | Every `consult_technical_manual` call |
| Pinecone | `vector_tools.py:20`, `ingest_manual.py:27` | Vector search / upsert | Every `consult_technical_manual` call |

Every one of the four is a paid or quota-limited third-party network service. The system has **no
offline mode**: with no network, the chat tab cannot produce any answer at all.

### 7.2 Environment variables

Four, all read via `os.getenv` after `load_dotenv()`. Names are listed in the local `.env`
(which is correctly `.gitignore`d) and in `.github/workflows/pipeline.yml:33-36`.

| Variable | Read at | Behaviour when missing — verified |
|---|---|---|
| `GOOGLE_API_KEY` | `main.py:59` | **Hard crash at startup.** `IndustrialAI()` raises `pydantic.ValidationError: Value error, API key required for Gemini Developer API...`. Since `app.py:20` constructs it at import time, the whole Streamlit page fails. |
| `TAVILY_API_KEY` | `search_tools.py:8` | **Depends on the installed version** — see §7.3. |
| `HUGGINGFACE_API_KEY` | `vector_tools.py:9` | **Silent degradation.** `get_vectorstore` raises `ValueError`, `search_manual` catches it, and the model receives `Error searching technical manual: HUGGINGFACE_API_KEY is missing from environment variables.` as if it were manual text. |
| `PINECONE_API_KEY` | `vector_tools.py:23` | **Silent degradation**, same path — the Pinecone client's own error is caught and stringified into the model's context. |

No variable has a default, none is validated at startup, and there is no `.env.example` to tell a new
user what is required. The only place the four names appear together is the CI workflow.

### 7.3 The unpinned Tavily dependency changes the failure mode

`requirements.txt` line 127 is bare `tavily-python`, no version. Verified consequence:

- The repository's own `venv` has **tavily-python 1.1.0**, whose `TavilyClient.__init__` ends with
  `if not self.api_key: raise MissingAPIKeyError(...)`. Because `search_tools.py:8` instantiates the
  client **at module import time**, and `app.py:5` imports `search_tools` at module level, a missing
  `TAVILY_API_KEY` under 1.1.0 crashes the entire application at startup — even for a user who only
  wants the dashboard and never touches web search.
- A fresh `pip install tavily-python` in a clean Python 3.12 environment on 2026-08-07 resolved to
  **0.7.27**, whose constructor accepts a `None` key and defers failure to call time. Verified: under
  0.7.27, `import tools.search_tools` succeeds with no key set.

So the same `requirements.txt` produces two materially different startup behaviours depending on when
you install it.

### 7.4 Deployment configuration

- **`Dockerfile`**: `python:3.11-slim`, installs `build-essential` and `gcc` (kept in the final image —
  no multi-stage build), `COPY . .`, exposes 8501, runs `streamlit run src/app.py`. It does **not**
  set `PYTHONPATH`, and relies on Streamlit's implicit `sys.path` insertion for `from main import ...`
  and `from tools... import ...` to resolve.
- **There is no `.dockerignore`.** `COPY . .` therefore copies everything present in the build context,
  including `venv/` (318 distributions), `data/` (48 MB), `logs/`, `.git/` and — for a local build —
  **`.env` with all four live API keys baked into an image layer**. The CI build context comes from a
  fresh `actions/checkout`, so the pushed `ghcr.io` image is not affected by the `.env` problem; but
  any developer running `docker build .` locally produces an image containing their secrets, and the
  `.git` directory and `venv` bloat every build.
- **`.github/workflows/pipeline.yml`**: on push/PR to `main`, installs requirements, runs `pytest tests/`
  with all four API keys injected from repository secrets, then logs in to GHCR and pushes
  `ghcr.io/lomadadharmateja/vulcan-industrial-os:latest`. There is no lint step, no build cache, no
  version tag other than `latest`, and the image is pushed on every PR run, not only on merges to main.
- **`.devcontainer/devcontainer.json`**: launches Streamlit with `--server.enableCORS false
  --server.enableXsrfProtection false`. Both protections are disabled.

### 7.5 Committed secrets

**No secret is committed.** Verified two ways: `.env` is listed in `.gitignore` and
`git log --all --diff-filter=A -- .env` returns nothing, so it was never added; and a pattern scan of
the full history (`git log --all -p` against Google `AIza…`, Tavily `tvly-…`, HuggingFace `hf_…`,
Pinecone `pcsk_…` and OpenAI `sk-…` shapes) produced zero matches. No key value is reproduced in this
document. The live `.env` at the repository root does hold four real-looking key values; it is
correctly ignored, but see the `.dockerignore` finding above for how it can still escape.

---

## 8. Interface layer

`src/app.py`, Streamlit 1.57.0. Single page, `layout="wide"`, with three tabs plus a sidebar and a
fragment.

**State management.** Everything is in `st.session_state`, initialised at module level with no locking:
- `st.session_state.agent` — one `IndustrialAI` per browser session, built on first script run
  (`app.py:19-20`). This carries the `ConversationBufferMemory`, so conversation state is per-session.
- `st.session_state.telemetry_results` — `calculate_risk_scores()` converted to a list of dicts
  (`app.py:22-25`). Computed once per session and **never refreshed**, despite the UI calling it
  "real-time".
- `st.session_state.messages` — the chat transcript rendered in the UI (`app.py:87-88`), a **second,
  independent** copy of the history alongside the agent's memory.

**Layout and triggers:**

| Element | Location | Triggers an agent/LLM call? |
|---|---|---|
| Sidebar checkbox "Enable Live Supply Chain Feed" | `app.py:33-35` | No LLM call, but calls **Tavily directly**, bypassing the agent, on every rerun while checked. Hardcoded query `"Global industrial logistics disruptions 2026"`; output truncated to 500 chars for display. |
| `@st.fragment(run_every="1d")` daily audit | `app.py:40-59` | Only via its "Generate Repair Strategy" button — which is unreachable (§5.9). |
| Dashboard tab: last 10 maintenance rows, commodity line chart | `app.py:65-70` | No |
| Dashboard tab: "🚀 Run Executive System Audit" button | `app.py:72-74` | **Yes** — `run_analysis()`, a 5-step multi-tool prompt |
| Predictive tab: five `st.metric` health tiles | `app.py:76-81` | No — reads cached `telemetry_results` |
| Chat tab: `st.chat_input` | `app.py:96-113` | **Yes** — `executor.invoke({"input": prompt})` |

**Concurrency and repeat invocation are not handled anywhere.** There is no in-flight guard, no
disabling of the buttons while a call is running, no idempotency key and no lock around the shared
`ConversationBufferMemory`. Streamlit serves each browser session on its own script thread, so two
sessions get two agents (fine), but within one session a user can trigger the audit button while a chat
turn is in flight; both mutate the same memory object. `run_analysis` additionally calls
`time.sleep(4)` after every successful invoke and `time.sleep(15)` on a rate-limit error
(`main.py:108, 115`), blocking that session's script thread for the duration with no UI feedback.

Further UI defects, all read directly from the code:

- **`app.py:70`** — the commodity chart is
  `pd.read_csv('data/commodity.csv').tail(20).set_index('date')['price_nominal_usd']`. Because the file
  is grouped by commodity (§2.2), `tail(20)` is the last 20 rows of the **Zinc** block (2023-05 to
  2024-12), not the most recent prices, and not a series the label or the dashboard identifies. The
  chart is unlabelled, so the user sees an anonymous line that is neither "latest" nor "all commodities".
  Both this line and `app.py:68` re-parse their CSVs (18.4 MB and 522 KB) on **every rerun**, with no
  `@st.cache_data`.
- **`app.py:113`** — `st.session_state.messages.append({"role": "assistant", "content": output})` sits
  *outside* the `with st.chat_message("assistant")` block. If `executor.invoke` raises, the user's
  message has already been appended at line 102 but `output` is never bound, so the transcript is left
  with a dangling user turn and the next rerun raises `NameError`. Combined with the fact that the
  agent's own memory *has* recorded the failed turn, the two histories drift apart.
- **`app.py:27-28`** — `clean_response()` is defined and never called. Dead code.
- **`app.py:37`** — the sidebar advertises "Brain: Gemini 2.5 Flash-Lite" and "Compute: Cloud
  Orchestrated". The first is factually wrong (§4.1); the second is not a technical statement.
- **`app.py:47, 81`** — unit labels are `Product ID[-5:]`, the last five characters of an ID like
  `M24859`, giving `24859`. Harmless but arbitrary.
- No spinner, error boundary or `st.exception` handling wraps any agent call except the audit button's
  internal `try` in `run_analysis`. A tool that raises (§5) renders a raw Python traceback in the browser.

---

## 9. End-to-end execution trace

### 9.1 A query that succeeds

**User types into the Chat tab: `What is wrong with machine L47230 and what will it cost to fix?`**

1. Streamlit reruns `src/app.py` top to bottom. `st.session_state.agent` already exists, so
   `IndustrialAI.__init__` does not run again. `telemetry_results` is cached. The sidebar renders; the
   Tavily checkbox is unchecked so no search fires. The fragment renders "All units operating within
   normal parameters" (§5.9). `app.py:68` re-parses `maintenance.csv`; `app.py:70` re-parses
   `commodity.csv` (18.4 MB) for the chart.
2. `app.py:96` — the walrus assignment captures the prompt. `app.py:98-99` echo it; `app.py:102`
   appends `{"role": "user", ...}` to `st.session_state.messages`.
3. `app.py:108` — `st.session_state.agent.executor.invoke({"input": prompt})`, entering
   `CleanAgentExecutor.invoke` → `AgentExecutor.invoke` → `AgentExecutor._call`.
4. `ConversationBufferMemory.load_memory_variables` returns `chat_history` (all prior turns, verbatim).
   `ChatPromptTemplate` renders: system prompt, chat history, the human turn, empty `agent_scratchpad`.
5. **External call 1** — `ChatGoogleGenerativeAI` POSTs to
   `generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:streamGenerateContent`
   with the 9 tool schemas attached, `temperature=0.2`. Gemini returns a tool call, most plausibly
   `check_maintenance_sensors(product_id="L47230")`.
6. `AgentExecutor` runs the tool. `data_tools.py:101` reads `data/maintenance.csv` from the current
   working directory. The row is found; `data.to_dict()` returns a dict of 14 columns — **including
   `Machine failure: 1` and `PWF: 1`**. LangChain's `_format_output` sees a non-string, calls
   `_stringify` → `json.dumps`, and wraps it in a `ToolMessage`. The agent has now been handed the
   ground-truth label (§4.7). The step is appended to `intermediate_steps`, and iteration count → 1
   of 15.
7. **External call 2** — the scratchpad now contains step 1; the model is called again. Following
   system-prompt rule 1 it calls `consult_technical_manual(query="power failure repair")`.
8. `data_tools.py:123` performs a **function-local import** of `tools.vector_tools`, which on first use
   imports `langchain_community` and `langchain_pinecone` — several seconds of import cost inside the
   request. `get_vectorstore()` reads `HUGGINGFACE_API_KEY`, constructs
   `HuggingFaceInferenceAPIEmbeddings`, constructs `PineconeVectorStore(index_name="vulcan-manuals")`.
9. **External calls 3 and 4** — `similarity_search(query, k=3)` sends the query text to the
   HuggingFace Inference API for a 384-dim embedding, then queries Pinecone. Three chunk texts come
   back and are joined with newlines. No page numbers, no scores. Iteration → 2.
10. **External call 5** — the model, following rule 2, calls
    `get_internal_commodity_prices(commodity_name="Copper")`. `data_tools.py:61` re-parses the entire
    18.4 MB `commodity.csv`, substring-matches, takes `.iloc[-1]`, and returns
    `Internal Record: Copper is 12986.6068 per ($/mt).` Iteration → 3.
11. **External call 6** — with three tool results in the scratchpad, Gemini emits a final answer as a
    list of content blocks.
12. `AgentExecutor` returns `{"input": ..., "chat_history": [...], "output": [...]}`.
    `CleanAgentExecutor.invoke` (`main.py:48-49`) replaces `response["output"]` with
    `clean_gemini_output(...)`, unwrapping `output[0]['text']` to a plain string.
13. `AgentExecutor` writes the turn into `ConversationBufferMemory` — which now permanently holds the
    full JSON of the machine row, the three manual chunks and the price string, to be resent on every
    subsequent turn for the rest of the session.
14. `app.py:110` renders the text; `app.py:113` appends it to `st.session_state.messages`. Streamlit
    reruns once more to paint the final state, re-parsing both CSVs again.

Net: 6 Gemini requests, 1 HuggingFace request, 1 Pinecone request, ~19 MB of CSV parsed at least twice,
one 6.9 MB pickle untouched (the ML model was never consulted), and a permanent context growth of
several kilobytes.

### 9.2 A query that fails

**User types: `Compare torque against vibration for machine M14860`**

1. Steps 1-4 as above.
2. **External call 1** — Gemini sees `analyze_sensor_trends(product_id, sensor_name)` and the word
   "vibration", and emits `analyze_sensor_trends(product_id="M14860", sensor_name="Vibration")`.
   Nothing in the tool schema or docstring enumerates the five valid sensor names, so this is the
   model behaving reasonably.
3. `data_tools.py:109` reads the CSV. `machine_data` is non-empty, so the guard at line 112 passes.
4. `data_tools.py:114` executes `df['Vibration'].mean()`. **Verified: raises
   `KeyError: 'Vibration'`.**
5. No `handle_tool_error` is configured on the tool, so `BaseTool.run` re-raises. `AgentExecutor` has
   no error handling for tool exceptions, so the `KeyError` propagates out of `executor.invoke`.
6. `app.py:108` is not wrapped in `try`, so the exception escapes the `with st.spinner(...)` block and
   Streamlit renders a red traceback box reading `KeyError: 'Vibration'` in the chat area.
7. `output` is never bound, so `app.py:113` never runs. But `app.py:102` already appended the user's
   message. `st.session_state.messages` now ends with a user turn and no assistant reply.
8. On the next rerun, `app.py:91-93` replays the transcript, showing the user's orphaned message with
   no answer and no error — the failure becomes invisible. Meanwhile the agent's
   `ConversationBufferMemory` never recorded the turn at all, so the two histories are now
   inconsistent.
9. The user retypes something; the agent has no memory of the failed attempt and may try the same
   nonexistent sensor again.

A close variant fails differently: `sensor_name="Type"` passes the `KeyError` check but raises
`TypeError: Cannot perform reduction 'mean' with string dtype` at the same line — verified. And a
third variant, `get_internal_commodity_prices` with a regex metacharacter, raises `re.error` from
inside pandas — verified (§5.3). All three reach the browser as raw tracebacks.

---

## 10. Gaps and risks

**Testing**
1. Two tests, 35 lines, for 630 lines of code. Verified both pass (in a sandboxed copy, with dummy
   keys, no network): `2 passed, 405 warnings in 6.55s`.
2. Neither test asserts anything about correctness. `test_prediction_logic` asserts only that the
   output string contains `"Failure Probability"` — it would pass if the model returned 0 % for every
   input. `test_app_startup` asserts the sidebar title renders.
3. No test covers any of the eight other tools, the SQL path, the retrieval path, the agent loop, the
   memory, or `calculate_risk_scores`.
4. The CI workflow injects all four production API keys into the test job, so any test that ever does
   touch the network will spend real quota on every push and PR.

**Error handling**
5. Six of the nine tools have no `try`/`except` at all. Any pandas or file error becomes a browser
   traceback (§9.2).
6. `run_sql_query` and `search_manual` catch `Exception` broadly and return the message **as a
   successful result**, so failures are laundered into the model's context as content. `search_manual`
   is the worst case: retrieval can be completely broken and the agent will keep answering.
7. `get_market_news` has no error handling whatsoever, and `response['results']` will `KeyError` on any
   non-standard Tavily response.
8. `main.py:112-118` catches everything from `run_analysis` and returns the raw `str(e)` to the UI —
   which can include internal paths and request details.

**Timeouts, retries, rate limits**
9. No timeout is set on any of the four external services (Gemini, Tavily, HuggingFace, Pinecone).
   Only the Tavily client has a default (60 s).
10. No retry or backoff anywhere. The rate-limit "handling" at `main.py:114-116` is a `time.sleep(15)`
    that blocks the Streamlit thread and then gives up, telling the user to try again. The committed
    log shows Gemini returning 503 three times in April, so this is a live condition.
11. `time.sleep(4)` after every successful audit (`main.py:108`) serves no stated purpose and adds
    4 s to every run.
12. `AgentExecutor` has `max_execution_time=None`, so a slow chain can run for as long as 15
    iterations of external calls take, with the UI stuck on a spinner.

**Security**
13. **Arbitrary SQL execution from LLM-controlled input** (§5.4), with `DROP TABLE` verified to
    succeed against a copy of the database while the tool reports an error. The connection is not
    read-only. The attack path is: chat box → LLM → `run_sql_query` → SQLite. Anyone who can type into
    the chat can attempt to destroy the database, and a prompt-injection payload inside a Tavily search
    result or a manual chunk could do the same without the user's involvement.
14. **Regex injection** via `str.contains(commodity_name)` (§5.3) — user text compiled as a regex.
    A catastrophic-backtracking pattern against 49,093 rows is a denial-of-service vector, and a
    malformed one is a crash.
15. **No `.dockerignore`** — a local `docker build .` bakes `.env` (four live keys), `.git/` and
    `venv/` into the image (§7.4).
16. **CORS and XSRF protection are explicitly disabled** in the devcontainer launch command.
17. The Docker image is pushed to a public-by-default GHCR tag on every PR, not only on merge.
18. No authentication, authorisation or rate limiting on the Streamlit app itself. Every visitor shares
    the same API keys and can spend the owner's Gemini/Tavily quota.
19. `build-essential` and `gcc` remain in the shipped image — unnecessary attack surface from a
    single-stage build.

**Unbounded growth and cost**
20. `ConversationBufferMemory` is unbounded (§4.5) and every tool result is retained verbatim forever.
21. `run_sql_query` can return an 8.5 MB string into that memory — verified.
22. `predict_failure` re-reads a 6.9 MB pickle on every call; `get_internal_commodity_prices` re-parses
    an 18.4 MB CSV on every call; `app.py` re-parses both CSVs on **every Streamlit rerun**, including
    every keystroke-triggered rerun. No `@st.cache_data` or `@st.cache_resource` anywhere.
23. The daily-audit fragment (`run_every="1d"`) will re-execute on a long-lived session; it is harmless
    today only because the alert branch is unreachable.

**Correctness**
24. **`health_score` is exactly 80.0 for all ten dashboard units, and the alert threshold is `< 80`**,
    so the entire critical-alert path is dead code in practice (§5.9). Verified.
25. **Feature order at inference is unenforced**, and getting it wrong changes a verified prediction
    from 0.00 to 0.70 (§3.3).
26. `get_failed_machines` mis-attributes failure modes for the 24 multi-flag rows and reports
    `RNF: 0` where the data has 19 (§5.2). Verified.
27. `get_internal_commodity_prices` reports the user's search term as if it were the matched commodity
    (§5.3). Verified.
28. `.iloc[-1]` is treated as "latest" on a file that is not date-sorted (§2.2, §5.3).
29. The dashboard chart plots an unlabelled Zinc series while presenting it as commodity prices (§8).
30. `check_maintenance_sensors` leaks the ground-truth failure labels to the model (§4.7).
31. The `category` column of `commodity.csv` labels Aluminum, Nickel, Zinc, Tin, Lead and Iron ore as
    `Fertilizers` (§2.2). Any agent reasoning that filters on `category` is reasoning on wrong data.
32. `Actual_Demand` in the raw supply-chain data is a lambda `repr`; the cleaned version is unseeded
    random jitter of the forecast (§2.3, §2.4).

**Reproducibility**
33. **No evaluation of the ML model exists** — no split, no metric, no baseline (§3.2).
34. The training table is built with unseeded noise, so the shipped `.pkl` cannot be reproduced (§2.5).
35. The 5× row inflation means the obvious evaluation would have been inflated by ~23 F1 points (§3.2).
    Verified.
36. `scikit-learn==1.8.0` is pinned but the model was pickled with 1.7.2 — verified
    `InconsistentVersionWarning` on load (§3.1).
37. **The Pinecone index cannot be created from this repository** (§6). Its configuration exists only
    in a Pinecone account.
38. `pypdf` is required but undeclared, so the ingestion script cannot run from a clean install (§1.3).
39. `tavily-python` is unpinned and the two resolvable versions differ in startup failure mode (§7.3).
40. `requirements.txt` is UTF-16, machine-generated, and 36 of its 149 pins are unreachable from any
    import (§1.3).
41. The repository's own `venv/` is committed to disk (though `.gitignore`d) and is broken — its base
    interpreter no longer exists.

**Hardcoded values**
42. Model name `gemini-3.1-flash-lite`, temperature `0.2`, Pinecone index `vulcan-manuals`, embedding
    model name (twice, in two files), chunk size `1000`/overlap `100`, `k=3`, risk thresholds
    `0.8`/`0.5`, supplier threshold `0.8` and `head(3)`, AI4I physical thresholds
    `8.6`/`1380`/`3500`/`9000`/`11000`, health cutoff `80`, sidebar query string
    `"Global industrial logistics disruptions 2026"`, DB inflation factor `5`, noise `N(0,2)`,
    all four data file paths (relative, cwd-dependent), `time.sleep(4)` and `time.sleep(15)`.
    There is no config file, no constants module and no CLI.

**Dead code and inconsistency**
43. `src/graph_logic.py` — an entire LangGraph module that is never imported and returns two hardcoded
    strings ("Bearings are overheating per WEG page 45.", "Copper is at an all-time high…"). It is the
    only consumer of the `langgraph` dependency.
44. `app.py:27-28` `clean_response()` — defined, never called.
45. `main.py:6` imports `HumanMessage, SystemMessage` — never used.
46. `main.py:4` and `search_tools.py:3` both call `load_dotenv()`; `vector_tools.py` calls neither and
    depends on import order.
47. The `query` parameters of `get_failed_machines` and `get_supplier_info` are accepted and ignored.
48. `search_tools.py:12, 15` use `print()` for diagnostics while the rest of the system uses `logging`.
49. The model identifier is stated three different ways across `main.py` (`3.1-flash-lite`), the log
    message in `main.py:55` (matches), `app.py:37` (`2.5 Flash-Lite`) and the README (`2.5 Flash-Lite`).
50. `logs/factory_brain.log` is committed and not ignored, so every run produces a spurious diff.

**Where it breaks under realistic conditions**
51. A user asks a broad question ("show me everything about failures") → the model writes
    `SELECT * FROM maintenance` → 8.5 MB into the context → the Gemini request fails, and the memory is
    now poisoned for the rest of the session.
52. The HuggingFace serverless endpoint retires this model, or the Pinecone free index is reclaimed for
    inactivity → every manual lookup silently returns an error string → the agent keeps answering
    repair questions with no grounding and no warning to anyone.
53. Anyone runs `db_setup.py` again → the model no longer matches the database it queries.
54. Two people open the app at once → two `IndustrialAI` instances → two sets of API calls on one key,
    with no rate limiting, against a Gemini free tier that the committed log already shows returning
    503 and 429.

---

## 11. Claims in the existing README

Every factual or performance claim in `README.md`, in order. Line numbers refer to the current file.

| # | Line | Claim | Verdict | Evidence |
|---|---|---|---|---|
| 1 | 1 | Repository is at `github.com/LomadaDharmateja/vulcan-industrial-os` | `UNVERIFIABLE` | No remote is configured in the local clone and no network check was made. The CI pushes to `ghcr.io/lomadadharmateja/vulcan-industrial-os`, which is consistent. |
| 2 | 4 | "VULCAN operates as an Agentic Loop … observes a problem, selects the right tool, and reasons through a solution" | **CONFIRMED** | `create_tool_calling_agent` + `AgentExecutor` with 9 tools and `max_iterations=15` is a genuine tool-calling loop. |
| 3 | 7 | "The core of the system is the **Gemini 2.5 Flash-Lite** model" | **CONTRADICTED** | `main.py:58` sets `model="gemini-3.1-flash-lite"`; `logs/factory_brain.log` shows live calls to that endpoint. Repeated at line 45. |
| 4 | 7 | "orchestrated via LangChain" | **CONFIRMED** | `langchain-classic==1.0.7` agent API. |
| 5 | 8 | "It calls `get_failed_machines` to identify specific error codes like PWF or HDF **from the SQL database**" | **CONTRADICTED** | `data_tools.py:34` reads `data/maintenance.csv`. The tool never touches SQLite. |
| 6 | 8 | The agent identifies PWF/HDF codes | **CONFIRMED with a caveat** | It does return those codes, but mis-attributes 24 multi-flag rows and never returns RNF (§5.2, verified). |
| 7 | 9 | "It checks the failure against `analyze_sensor_trends` to see if **real-time telemetry** … confirms the mechanical stress" | **CONTRADICTED** | The data is a static CSV with one row per unit (all 10,000 `Product ID`s unique) and a file date of 2022-11-06. There is no telemetry and no time dimension; "trend" is one row compared to a global mean. |
| 8 | 10 | "It performs a Vector Search in `vector_tools.py` to pull specific repair steps from indexed technical manuals" | **CONFIRMED (mechanism) / CONTRADICTED (plural)** | The code does exactly this. But there is exactly **one** manual, 54 pages. |
| 9 | 11 | "It consults `get_supplier_info` and `get_market_news` to provide a logistical plan" | **CONFIRMED** | Both tools are registered and reachable. |
| 10 | 14 | "VULCAN is built for **Edge Deployment**" | **CONTRADICTED** | Every inference path requires four external cloud APIs. With no network the chat tab cannot answer at all. Nothing about this system runs at an edge. |
| 11 | 15 | "HuggingFace Inference API: Text chunks … are converted to vectors **in the cloud, keeping the local server lightweight**" | **CONFIRMED (mechanism) / misleading (framing)** | The API call is real. It removes local RAM cost by adding a network dependency, a per-call latency and an auth failure mode that degrades silently (§5.7). |
| 12 | 16 | "Pinecone Serverless" | `UNVERIFIABLE` | The index tier, dimension, metric and region are not in the repository and no Pinecone call was made. Only the index name `vulcan-manuals` is in the code. |
| 13 | 16 | "allowing the AI to find specific repair paragraphs **in milliseconds**" | `UNVERIFIABLE` | No latency measurement exists in the repo and none was taken. The round trip is HF embedding + Pinecone query over the public internet, plus a cold import of `langchain_pinecone` on first use. |
| 14 | 19 | "uses a Random Forest Classifier trained on industrial data" | **CONFIRMED, with the origin misstated** | Verified `RandomForestClassifier`, 100 trees. The data is the synthetic AI4I dataset (§2.1), inflated 5× (§2.5) — not industrial data from any plant. |
| 15 | 20 | "It analyzes the relationship between rotational speed, torque, and tool wear to predict failures **before they happen**" | **CONTRADICTED** | The model is a point classifier on a single row of five instantaneous readings. It has no time dimension, no horizon and no lead time. It classifies the present, it does not forecast. |
| 16 | 21 | "`predict_failure` returns a probability percentage, allowing operators to intervene during STABLE or WARNING phases" | **CONFIRMED (mechanism) / UNVERIFIABLE (usefulness)** | The string format is exactly as described. But the 0.5/0.8 thresholds are arbitrary and uncalibrated, and no evaluation exists to say what they mean (§3). |
| 17 | 27 | Telemetry from `maintenance.csv` used for "real-time monitoring" | **CONTRADICTED** | Static file, no ingestion, no refresh; `telemetry_results` is computed once per session and cached (§8). |
| 18 | 28 | Logistics from `industrial_cleaned.csv` for "supplier reliability and lead-time optimization" | **CONFIRMED (source) / CONTRADICTED (optimization)** | The file is read. There is no optimisation — `get_supplier_info` is a filter, a sort and a `head(3)`. |
| 19 | 29 | Market: "Internal tracking of material costs (**Copper/Steel**)" | **CONTRADICTED for Steel** | Verified: 0 of 49,093 rows match "Steel" across all 71 commodities. `get_internal_commodity_prices("Steel")` returns "Material not found in local records." Copper is present (793 rows). |
| 20 | 30 | Manuals: "Grounded technical instructions via Pinecone RAG" | **CONFIRMED (mechanism)** | But retrieved chunks carry no page or source citation, so groundedness is unverifiable by the user (§5.7). |
| 21 | 33 | "Executive System Audit: A one-click autonomous report that synthesizes technical, financial, and logistical data" | **CONFIRMED** | `app.py:72-74` → `run_analysis()` with a 5-step prompt spanning those tools. |
| 22 | 34 | "Daily Autonomous Audit: A proactive alert system that uses `st.fragment` to monitor fleet health" | **CONTRADICTED in effect** | The fragment exists and is declared `run_every="1d"`, but the alert branch is gated on `health_score < 80` and the score is exactly 80.0 for all ten monitored units — verified. No alert can ever fire on this data. |
| 23 | 35 | "Live Supply Chain Feed: Integration with Tavily AI to overlay global market news" | **CONFIRMED** | `app.py:33-35`, hardcoded query, output truncated to 500 characters for display. |
| 24 | 42 | "modular **Manager-Worker** architecture … scalable, easy to debug, and professionally structured" | `UNVERIFIABLE` (as stated) / partly **CONTRADICTED** | The module split is real and reasonable. "Scalable" is contradicted by unbounded memory, no caching, per-call model reloads and no concurrency handling. |
| 25 | 46 | "**Self-Correction:** The agent uses a `ConversationBufferMemory` to remember previous diagnostic steps" | **CONFIRMED (memory) / CONTRADICTED (self-correction)** | The memory is real and unbounded. There is no self-correction mechanism: no reflection step, no retry, no validation, and `handle_parsing_errors` is left `False`. |
| 26 | 47 | "Instead of hard-coded IF-THEN statements, the agent uses logic to decide whether it needs to query the SQL database … or search the Vector Store" | **CONFIRMED** | Tool selection is genuinely model-driven. |
| 27 | 51 | "`predict_failure` … loads a pre-trained `.pkl` model to provide real-time risk assessments based on **live telemetry**" | **CONTRADICTED** | The inputs are five numbers supplied by the LLM in the tool call. There is no telemetry source of any kind. |
| 28 | 52 | "the system can perform **complex cross-table joins** to link machine failures with supplier lead times" | **UNVERIFIABLE / misleading** | `run_sql_query` can execute a join if the model writes one, but the three tables share **no key** — `maintenance` has no supplier column and `logistics` has no machine column. Any join between them is a cross product, not a link. |
| 29 | 53 | "Standardized functions fetch internal commodity data and external market news … providing a 360-degree financial view" | **CONTRADICTED (standardized)** | The two functions differ in return format, error handling (one has none) and data source. "360-degree financial view" is unmeasurable. |
| 30 | 57 | "Semantic Search: … If an agent asks about 'excessive vibration,' the vector engine finds the corresponding troubleshooting section" | `UNVERIFIABLE` | Requires live HF + Pinecone calls, which were not made. The mechanism is present; retrieval quality has never been measured, and there is no evaluation set. |
| 31 | 58 | "the system can perform high-dimensional math **without requiring a local GPU**" | **CONFIRMED, trivially** | True, because the math happens on someone else's machine. all-MiniLM-L6-v2 is a 22M-parameter model that runs on CPU in any case. |
| 32 | 61 | "the UI is designed for high-pressure industrial environments" | **CONTRADICTED** | No auth, no error boundaries, raw tracebacks on tool failure (§9.2), blocking `sleep` calls, a dead alert path, and a mislabelled chart. |
| 33 | 62 | "Uses `st.session_state` to maintain a seamless chat experience and ensure data doesn't get lost when the user toggles between tabs" | **CONFIRMED (mechanism) / CONTRADICTED (seamless)** | State is kept, but a tool exception leaves an orphaned user message and an unbound `output`, desynchronising the two histories (§8, §9.2). |
| 34 | 63 | "`st.fragment` … allows the system to monitor critical risks **in the background without interrupting** the user's active analysis" | **CONTRADICTED** | Nothing is monitored (claim 22), and the "Generate Repair Strategy" button inside the fragment blocks the session thread when clicked. |
| 35 | 67 | "**Reduced Downtime:** Shift from Run-to-Failure to Predict-and-Prevent" | **CONTRADICTED / UNVERIFIABLE** | This system has never been connected to a machine. The model has no evaluation at all, and cross-validation on the source data gives recall 0.646 — it would miss roughly a third of failures. No downtime claim can be supported. |
| 36 | 68 | "**Knowledge Retention:** Junior technicians can access the expertise of senior engineers through AI-powered manual search" | `UNVERIFIABLE` | The knowledge base is one 54-page vendor manual with no citations returned. |
| 37 | 69 | "**Supply Chain Resilience:** Automated monitoring of commodity prices allows for smarter procurement during market dips" | **CONTRADICTED** | Nothing is monitored — there is no scheduler, no alert and no threshold on prices. The data ends at 2026-02-01 and is a static file. |
| 38 | throughout | The `[cite: N]` markers on lines 4, 7, 8, 9, 19, 20, 21, 27, 29, 30, 33, 34 | **CONTRADICTED** | These are artifacts of a document-generation tool. They reference sources that do not exist anywhere in the repository. |

**Summary of the README audit:** 38 claims examined — 8 CONFIRMED, 6 CONFIRMED with a material caveat,
17 CONTRADICTED, 7 UNVERIFIABLE. The recurring pattern is that mechanisms described in the README do
exist in the code, while every adjective attached to them ("real-time", "edge", "live", "monitoring",
"scalable", "optimization") does not survive contact with the implementation.

---

## Appendix: reproducing the measurements in this audit

Nothing in this appendix was run against the working tree; `data/` and `models/` were copied to a
scratch directory first, and the destructive-SQL test ran against that copy.

- Model inspection: `joblib.load` under scikit-learn 1.7.2 and again under 1.8.0, reading
  `get_params()`, `feature_names_in_`, `n_features_in_`, and per-tree `get_depth()`/`get_n_leaves()`.
- Pickle provenance: raw byte scan of the first 100 KB of `failure_predictor.pkl` for the
  `_sklearn_version` marker, corroborated by the presence/absence of `InconsistentVersionWarning`
  under the two runtimes.
- Data profiling: `pandas.read_csv` for all four CSVs; `sqlite3` + `pandas.read_sql_query` for the DB.
- Reference metrics: `StratifiedKFold(5, shuffle=True, random_state=0)` with `cross_val_predict`,
  same hyperparameters as `train_model.py`, on the original 10,000 rows; and `train_test_split`
  (80/20, stratified) on the inflated 50,000-row table for the leakage comparison.
- Chunk count: `PyPDFLoader` + `RecursiveCharacterTextSplitter(1000, 100)` run locally with no
  embedding or upsert step.
- Tool behaviour: each tool invoked via `.invoke({...})` with valid, invalid and adversarial inputs
  inside a sandbox whose working directory contained the copied `data/` and `models/`.
- Startup behaviour: `IndustrialAI()` constructed with and without `GOOGLE_API_KEY`; `tests/` run under
  `pytest` with four placeholder key values. No request left the machine.
- Dependency reachability: `Requires-Dist` graph walked from the 16 directly-imported distributions,
  parsed with `email.parser` from the `*.dist-info/METADATA` files in the repository's `venv`.
