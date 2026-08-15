# CLAUDE.md

## Purpose of this session

This repository contains an agentic AI system for industrial predictive maintenance that I built some time ago. I have lost track of the implementation details.

Your task is a **read-only forensic audit** of this codebase, followed by two written documents. You are not here to improve, refactor, or fix anything.

---

## Ground rules — read these before doing anything

1. **Read-only.** Do not modify, refactor, reformat, delete, or create any source file. The only files you may write are the two deliverables named at the bottom of this document.
2. **Verify, do not infer.** Where it is safe to do so, actually execute things to confirm your findings: load the CSVs and report real shapes, unpickle the model and inspect it, parse the dependency files. Do not describe behaviour you have only read in code comments or in the existing README.
3. **Mark uncertainty explicitly.** If you cannot confirm something, write `UNVERIFIED:` followed by what you believe and why you could not confirm it. Never fill a gap with a plausible guess.
4. **Do not run anything that costs money or hits an external API.** No calls to Gemini, Pinecone, HuggingFace Inference, or Tavily. Static analysis and local data inspection only.
5. **Do not print or copy any secret.** If you find API keys committed in the code, report the file and line number and the fact of exposure — never the value.
6. **No marketing language.** Plain technical prose throughout. No emoji, no adjectives like "powerful", "seamless", "production-grade", "enterprise", "cutting-edge". No business-impact claims.
7. **Treat the existing README as a claim to be tested, not as a source of truth.** It was written loosely and parts of it are known to be inaccurate.

---

## Deliverable 1 — `docs/PROJECT_AUDIT.md`

This is the raw, honest inventory. It is for my eyes and my advisor's, not for the public. Prioritise completeness and accuracy over readability.

Cover the following sections:

### 1. Repository inventory
Full file tree excluding `.git`, virtualenvs, caches, and data blobs. For each source module: line count and a one-line statement of responsibility. List every declared dependency and mark which are actually imported anywhere versus which are unused.

### 2. Data assets
For every CSV, PDF, database file, and serialised artifact in the repo:
- Exact filename, size, and location
- For tabular data: row count, column names with dtypes, null counts per column, and the value distribution of any target or label column
- **Explicitly report class balance** for any classification target
- Whether the data appears to be a known public dataset — name your best guess and mark it `UNVERIFIED` unless a source is documented in the repo
- Whether any data-generation, cleaning, or ingestion script exists, or whether the files simply appear as committed artifacts

### 3. Machine learning component
- Locate the serialised model. Load it and report: exact class, library and version it was pickled with, all hyperparameters, expected feature names and their order, number of features
- Locate the training script or notebook. If none exists, state that plainly and note that the model is therefore **not reproducible from this repository**
- Report any evaluation code or metrics found. If none exist, say so
- Report how the model is invoked at inference time and whether feature ordering at inference is guaranteed to match training

### 4. Agent layer
- Exact model identifier and provider
- Agent framework and version, and which agent construction pattern is used
- The system prompt, reproduced verbatim in a code block
- Complete list of registered tools with their function signatures and docstrings exactly as written
- Memory mechanism, and whether it is bounded
- Iteration limits, temperature, and any other generation parameters
- Whether tool outputs are validated before being returned to the model

### 5. Tool implementations
For each tool, one subsection covering: what it does, what data source it touches, any SQL reproduced verbatim, what it returns and in what format, what input validation exists, and what happens on failure.

### 6. Retrieval layer
Embedding model and dimensionality, vector store and index configuration, chunking strategy with exact parameters, how many chunks exist, which source documents were indexed, and whether a reproducible ingestion script exists or the index was populated manually.

### 7. External services and configuration
Every external API called. Every environment variable required. What the application does when each is missing or invalid — crash, silent failure, or graceful degradation.

### 8. Interface layer
Pages or tabs, state management approach, which user actions trigger agent invocations, and whether concurrent or repeated invocations are handled.

### 9. End-to-end execution trace
Pick one realistic user query. Trace it through the entire system step by step — every function entered, every external call, every state mutation — until the response is rendered. Then do the same for one query that would fail, and describe exactly how it fails.

### 10. Gaps and risks
An unvarnished list. Include at minimum: absence of tests, absence of error handling, absence of retries or timeouts, hardcoded values, committed secrets, SQL injection surface, unbounded memory growth, missing input validation, dead code, silent exception swallowing, and any place where the system would break under realistic conditions.

### 11. Claims in the existing README that you could not verify
Go through the current README line by line. List every factual or performance claim, and mark each as `CONFIRMED`, `CONTRADICTED`, or `UNVERIFIABLE`, with the evidence.

---

## Deliverable 2 — `docs/README.draft.md`

Do **not** overwrite the existing `README.md`. Write a new draft at the path above.

This should read like an engineering design document, not a product page. Structure:

1. **What this is** — two or three sentences, plain description
2. **Problem statement and scope** — including an explicit *non-goals* subsection
3. **Data** — dataset name, source, and licence if determinable; what it contains; **its limitations, stated openly** (synthetic origin, size, class imbalance, single-document knowledge base — whatever is true)
4. **Architecture** — component diagram in Mermaid, followed by prose explaining the flow
5. **Tool reference** — table of every agent tool: name, purpose, data source, inputs, outputs
6. **Machine learning model** — what it predicts, features used, how it was trained, and a metrics section left as `TODO: not yet measured` where no evaluation exists
7. **Setup and running** — prerequisites, installation, required environment variables, how to launch
8. **Configuration reference** — every env var and config value with its purpose and default
9. **Known limitations and failure modes** — carry the honest findings from the audit through to here
10. **Roadmap** — what is missing and would need to exist for this to be considered production-ready

### Hard constraints on the draft README
- Every number must be one you actually measured or found documented in the repo. If it has not been measured, write `TODO: not yet measured` rather than estimating.
- No claims about business outcomes, cost savings, downtime reduction, or operational impact. This project has never touched a real factory.
- No claim that the system is production-ready, enterprise-grade, or edge-deployable.
- If a component depends on an external network service, say so plainly in the architecture section rather than describing it as lightweight or local.

---

## When you are done

Finish with a short summary in the chat covering:
- The three findings that most surprised you
- The three things a hiring engineer reviewing this repo would criticise first
- Anything in the codebase that is genuinely well done and worth keeping

Then stop. Do not begin implementing fixes.