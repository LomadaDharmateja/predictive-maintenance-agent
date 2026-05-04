git repository [https://github.com/LomadaDharmateja/vulcan-industrial-os]
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
