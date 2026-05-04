import streamlit as st
import pandas as pd
from main import IndustrialAI
from tools.data_tools import calculate_risk_scores
from tools.search_tools import get_live_market_news

# --- PAGE CONFIG ---
st.set_page_config(page_title="VULCAN AI | Industrial OS", layout="wide", page_icon="⚡")

# --- CSS (Kept your modern dark theme) ---
st.markdown("""
    <style>
    [data-testid="stMetric"] { background-color: #1E293B !important; border: 1px solid #334155 !important; border-radius: 12px !important; }
    .main-header { font-size: 2.5rem; font-weight: 600; color: #F8F9FA; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZATION ---
if "agent" not in st.session_state:
    st.session_state.agent = IndustrialAI()

if "telemetry_results" not in st.session_state:
    with st.spinner("Analyzing Fleet Health..."):
        risk_df = calculate_risk_scores()
        st.session_state.telemetry_results = risk_df.to_dict('records')

def clean_response(output):
    return str(output) # Simplified for standard agent output

# --- SIDEBAR ---
with st.sidebar:
    st.title("System Controls")
    if st.checkbox("Enable Live Supply Chain Feed"):
        alerts = get_live_market_news("Global industrial logistics disruptions 2026")
        st.caption(alerts[:500] + "...")
    st.markdown("---")
    st.info("**Compute:** Cloud Orchestrated\n\n**Brain:** Gemini 2.5 Flash-Lite")

# --- DAILY AUDIT (TOP ALERT) ---
@st.fragment(run_every="1d")
def autonomous_daily_audit():
    st.markdown("### 📅 Daily Autonomous Audit")
    results = st.session_state.telemetry_results
    top_issue = min(results, key=lambda x: x['health_score'])

    if top_issue['health_score'] < 80:
        st.error(f"🚨 **CRITICAL: Unit {top_issue['Product ID'][-5:]} at High Risk**")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Health", f"{int(top_issue['health_score'])}%", delta="-CRITICAL")
        with col2:
            if st.button("Generate Repair Strategy", type="primary"):
                with st.spinner("Consulting Brain..."):
                    res = st.session_state.agent.executor.invoke({"input": f"Create repair plan for {top_issue['Product ID']}"})
                    st.markdown(res["output"])
    else:
        st.success("✅ All units operating within normal parameters.")

autonomous_daily_audit()

# --- MAIN UI ---
st.markdown('<h1 class="main-header">⚡ VULCAN Industrial OS</h1>', unsafe_allow_html=True)
tab_dash, tab_pred, tab_chat = st.tabs(["📊 Dashboard", "🔮 Predictive", "💬 Chat"])

with tab_dash:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.dataframe(pd.read_csv('data/maintenance.csv').tail(10), use_container_width=True)
    with col_right:
        st.line_chart(pd.read_csv('data/commodity.csv').tail(20).set_index('date')['price_nominal_usd'])
    
    if st.button("🚀 Run Executive System Audit"):
        report = st.session_state.agent.run_analysis()
        st.info(report)

with tab_pred:
    risk_df = pd.DataFrame(st.session_state.telemetry_results)
    cols = st.columns(5)
    for i, (_, row) in enumerate(risk_df.iterrows()):
        with cols[i % 5]:
            st.metric(f"Unit {row['Product ID'][-5:]}", f"{int(row['health_score'])}%")

with tab_chat:
    st.markdown("### Consult System Engineer")
    
    # 1. Initialize history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. ALWAYS display the history first (This keeps the order straight)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Handle the new input
    if prompt := st.chat_input("Ask about maintenance, manuals, or parts..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to session state history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Orchestrating Agents..."):
                # Call the executor
                response = st.session_state.agent.executor.invoke({"input": prompt})
                output = response["output"]
                st.markdown(output)
        
        # Add assistant response to session state history so it stays on rerun
        st.session_state.messages.append({"role": "assistant", "content": output})