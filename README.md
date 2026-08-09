# Industrial AI agent — predictive maintenance planning

git repository [https://github.com/LomadaDharmateja/vulcan-industrial-os]

## What this system is for

> **Flag elevated component risk over a 14-day window so maintenance attention can be
> scheduled, and manage parts from stock levels and consumption rates rather than from
> predictions.**

**This replaces the original predict-and-order framing.** The change was forced by
measurement.

The project was built to predict a failure and then order the replacement part. That
cannot work on this data. The model's effective detection lead — the gap between its
score crossing the operating threshold and the failure — is a median of about 14 days
for comp2, comp3 and comp4, and about 24 hours for comp1. Supplier lead times run 10 to
34 days, median 23. Crossing the two lists, **1 of 9 parts can be ordered in time.**

Extending the horizon does not rescue it. Predictability caps the horizon at 14 days;
beyond that the model's bootstrap interval overlaps a matched-error-code baseline's and
it is no longer established as better than a spreadsheet rule. The lead-time
requirement starts at 23 days. The two constraints do not intersect, and no horizon
satisfies both. Derivation in [`docs/SIGNAL_ANALYSIS.md`](docs/SIGNAL_ANALYSIS.md)
section 4.

So the system does two separate things, and keeping them separate is the design:

| It does | It does not |
|---|---|
| Flag which component on which machine is at elevated risk over 14 days, as a calibrated probability with a confidence interval | Say when inside that window, or how severe |
| State whether its warning is long enough to act on, per component and per part | Recommend ordering a part on the strength of a prediction |
| Report parts position from stock on hand and observed consumption | Derive a reorder decision from a risk score |

The agent layer expresses that separation in the type system: `get_parts_position`
accepts no risk score and has no import path to the model, asserted by walking the
import graph in `tests/test_agent_parts_independence.py`.

### Where the detail lives

| Document | Contents |
|---|---|
| [`docs/DATA.md`](docs/DATA.md) | Dataset, schema, leakage risks, the horizon decision |
| [`docs/FEATURES.md`](docs/FEATURES.md) | The 38 features, the labels, the splits |
| [`docs/SIGNAL_ANALYSIS.md`](docs/SIGNAL_ANALYSIS.md) | Fault-signature lead times, horizon sweep, the negative result |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Baselines, metrics, calibration, thresholds, which model ships |
| [`docs/EVALUATION_24h.md`](docs/EVALUATION_24h.md) | The archived 24-hour evaluation and why it was useless |
| [`docs/leakage-case-study.md`](docs/leakage-case-study.md) | What v1 got wrong, recomputed from the archived code |


## Getting the data

The five Azure PdM source files are not committed. `PdM_telemetry.csv` alone is 77 MB,
and git history cannot be trimmed back afterwards without a rewrite. Fetch them
instead:

```
make fetch-data          # download into data/raw/ and verify every checksum
make fetch-data-verify   # verify what is already on disk; downloads nothing
```

Each file is checked against a SHA-256 recorded in `scripts/fetch_data.py`. A
truncated download or a silently re-uploaded upstream dataset fails here rather than
surfacing later as an unexplained change in a measurement.

### Kaggle credentials

The download needs a Kaggle account. `scripts/fetch_data.py` never reads, stores, logs
or prints a credential — it hands authentication to the `kaggle` library, which
resolves it itself. Set it up one of these ways, in the shell, not in this repository:

| Method | How |
|---|---|
| OAuth (recommended) | `kaggle auth login` — opens a browser, caches a token outside the repo |
| Token in the environment | Generate a token at <https://www.kaggle.com/settings/api>, then `export KAGGLE_API_TOKEN=...` |
| Token file | Save the same token to `~/.kaggle/access_token` |
| Legacy key pair | `~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` and `KAGGLE_KEY` |

Never commit a token, and never place one inside this repository. `.env` is gitignored,
but nothing in this project reads Kaggle credentials from it, so putting one there
would give you a secret on disk and no working download.

If you would rather not use the API at all, download the archive by hand from
<https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance>,
unzip the five `PdM_*.csv` files into `data/raw/`, and run `make fetch-data-verify` to
confirm they are the expected ones.

See `docs/DATA.md` for what the data contains and its limitations.

---

> **Everything below this line describes the archived v1 of this project
> (`archive/v1-app/`, `archive/v1-data/`) and does not describe the current codebase.
> Several of its claims were contradicted by measurement; see
> `docs/v1/PROJECT_AUDIT.md` section 11.**

## 🧠 System Architecture & Operational Logic

VULCAN operates as an **Agentic Loop**, meaning it doesn't just follow a script; it observes a problem, selects the right tool, and reasons through a solution[cite: 3].

### 1. Autonomous Agentic Workflow
The core of the system is the **Gemini 2.5 Flash-Lite** model orchestrated via LangChain[cite: 3]. When a failure is detected, the agent follows a multi-step cognitive path:
* **Perception:** It calls `get_failed_machines` to identify specific error codes like **PWF** (Power Failure) or **HDF** (Heat Dissipation Failure) from the SQL database[cite: 3, 4].
* **Reasoning:** It checks the failure against `analyze_sensor_trends` to see if real-time telemetry (Torque, Temperature) confirms the mechanical stress[cite: 3, 4].
* **Knowledge Retrieval:** It performs a **Vector Search** in `vector_tools.py` to pull specific repair steps from indexed technical manuals.
* **Action:** It consults `get_supplier_info` and `get_market_news` to provide a logistical plan for replacement parts.

### 2. Cloud-Optimized RAG (Retrieval-Augmented Generation)
Unlike traditional RAG systems that require heavy local RAM to run embedding models, VULCAN is built for **Edge Deployment**:
* **HuggingFace Inference API:** Text chunks from manuals are converted to vectors in the cloud, keeping the local server lightweight.
* **Pinecone Serverless:** Stores technical motor manuals in a high-dimensional vector space, allowing the AI to find specific repair paragraphs in milliseconds.

### 3. Predictive Intelligence
The system uses a **Random Forest Classifier** trained on industrial data to provide "Future-Sight"[cite: 4]:
* It analyzes the relationship between rotational speed, torque, and tool wear to predict failures before they happen[cite: 4].
* The `predict_failure` tool returns a probability percentage, allowing operators to intervene during "STABLE" or "WARNING" phases rather than waiting for "CRITICAL" failure[cite: 4].

## 🛠️ Data Structure & Integration

| Component | Source | Usage in VULCAN |
|-----------|--------|-----------------|
| **Telemetry** | `maintenance.csv` | Real-time monitoring and sensor trend analysis[cite: 4]. |
| **Logistics** | `industrial_cleaned.csv` | Supplier reliability and lead-time optimization. |
| **Market** | `commodity.csv` | Internal tracking of material costs (Copper/Steel)[cite: 4]. |
| **Manuals** | `WEG-Manual.pdf` | Grounded technical instructions via Pinecone RAG[cite: 5]. |

## 📊 Dashboard Capabilities
* **Executive System Audit:** A one-click autonomous report that synthesizes technical, financial, and logistical data[cite: 3].
* **Daily Autonomous Audit:** A proactive alert system that uses `st.fragment` to monitor fleet health without refreshing the entire page[cite: 2].
* **Live Supply Chain Feed:** Integration with **Tavily AI** to overlay global market news on top of internal factory data.




## 🔧 Engineering Implementation & Code Logic

VULCAN is built on a modular "Manager-Worker" architecture. This separation of concerns ensures the system is scalable, easy to debug, and professionally structured.

### 1. The "Brain" (src/main.py)
The `IndustrialAI` class serves as the central command. It initializes the **Gemini 2.5 Flash-Lite** model and binds it to a suite of specialized tools. 
* **Self-Correction:** The agent uses a `ConversationBufferMemory` to remember previous diagnostic steps, allowing for complex, multi-turn troubleshooting.
* **Tool Orchestration:** Instead of hard-coded IF-THEN statements, the agent uses logic to decide whether it needs to query the SQL database for sensor data or search the Vector Store for repair guides.

### 2. The "Worker Tools" (src/tools/data_tools.py)
This module contains the functional logic that interacts with the factory's physical and digital assets.
* **Machine Learning Inference:** The `predict_failure` tool loads a pre-trained `.pkl` model to provide real-time risk assessments based on live telemetry.
* **Database Interfacing:** Using `sqlite3` and `pandas`, the system can perform complex cross-table joins to link machine failures with supplier lead times.
* **Market Intelligence:** Standardized functions fetch internal commodity data and external market news via the Tavily API, providing a 360-degree financial view.

### 3. The "Knowledge Base" (src/tools/vector_tools.py)
This is the Retrieval-Augmented Generation (RAG) engine.
* **Semantic Search:** Unlike a simple keyword search, this module understands the *context* of an error. If an agent asks about "excessive vibration," the vector engine finds the corresponding troubleshooting section in the motor manual.
* **Cloud-Native Design:** By leveraging the HuggingFace Inference API, the system can perform high-dimensional math without requiring a local GPU.

### 4. The "User Interface" (src/app.py)
Built with Streamlit, the UI is designed for high-pressure industrial environments.
* **State Management:** Uses `st.session_state` to maintain a seamless chat experience and ensure data doesn't get lost when the user toggles between the Dashboard and the Predictive tabs.
* **Performance Optimization:** Implements `st.fragment` for the Daily Audit, allowing the system to monitor critical risks in the background without interrupting the user's active analysis.

## 🚀 Impact & Use Cases

* **Reduced Downtime:** Shift from "Run-to-Failure" to "Predict-and-Prevent."
* **Knowledge Retention:** Junior technicians can access the expertise of senior engineers through the AI-powered manual search.
* **Supply Chain Resilience:** Automated monitoring of commodity prices allows for smarter procurement of replacement parts during market dips.
