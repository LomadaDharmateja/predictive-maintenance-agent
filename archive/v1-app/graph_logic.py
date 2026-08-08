from langgraph.graph import StateGraph, END

# Define the 'State' (what the agents share)
class AgentState(dict):
    messages: list
    technical_report: str
    financial_report: str

# 1. Technical Node: Checks Sensors & Manuals
def technical_expert(state):
    # Logic to call check_maintenance_sensors + consult_technical_manual
    return {"technical_report": "Bearings are overheating per WEG page 45."}

# 2. Financial Node: Checks Commodity Prices & News
def financial_analyst(state):
    # Logic to call check_market_prices + Tavily Search
    return {"financial_report": "Copper is at an all-time high; delay purchase if possible."}

# 3. Construct the Graph
workflow = StateGraph(AgentState)
workflow.add_node("tech", technical_expert)
workflow.add_node("finance", financial_analyst)

workflow.set_entry_point("tech")
workflow.add_edge("tech", "finance")
workflow.add_edge("finance", END)

app = workflow.compile()