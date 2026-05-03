import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from langchain.tools import tool

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'failure_predictor.pkl')

# --- ML INFERENCE TOOL ---
@tool
def predict_failure(air_temp: float, process_temp: float, speed: float, torque: float, tool_wear: float):
    """
    Predicts the probability of a machine failure based on sensor inputs.
    Inputs: Air Temp (K), Process Temp (K), Speed (rpm), Torque (Nm), Tool Wear (min).
    """
    if not os.path.exists(MODEL_PATH):
        return "Failure prediction model not found. Ensure models/failure_predictor.pkl exists."
    
    model = joblib.load(MODEL_PATH)
    features = [[air_temp, process_temp, speed, torque, tool_wear]]
    probability = model.predict_proba(features)[0][1]
    
    risk_level = "CRITICAL" if probability > 0.8 else "WARNING" if probability > 0.5 else "STABLE"
    return f"Failure Probability: {probability:.2%}. Risk Status: {risk_level}."

# --- DATA ANALYSIS TOOLS ---

@tool
def get_failed_machines(query: str = ""):
    """Returns a list of failed machines from maintenance.csv. Input is optional."""
    df = pd.read_csv('data/maintenance.csv')
    failures = df[df['Machine failure'] == 1]
    result = []
    for _, row in failures.iterrows():
        result.append({"product_id": row['Product ID'], "failure_type": row.get('failure_type', 'Unknown')})
    return str(result)

@tool
def get_internal_commodity_prices(commodity_name: str):
    """Checks the LOCAL commodity.csv for the last recorded price of a material."""
    df = pd.read_csv('data/commodity.csv')
    target = df[df['commodity_name'].str.contains(commodity_name, case=False)]
    if not target.empty:
        latest = target.iloc[-1]
        return f"Internal Record: {commodity_name} is {latest['price_nominal_usd']} per {latest['unit']}."
    return "Material not found in local records."

@tool
def get_supplier_info(query: str = ""):
    """Checks the internal logistics database for reliable suppliers. Input is optional."""
    df = pd.read_csv('data/industrial_cleaned.csv')
    best_suppliers = df[df['Reliability_Score'] > 0.8].sort_values(by='Lead_Time_Supplier')
    return str(best_suppliers[['Supplier_ID', 'Lead_Time_Supplier', 'Reliability_Score']].head(3).to_dict())

def calculate_risk_scores():
    """Calculates fleet health scores for the Daily Autonomous Audit dashboard."""
    df = pd.read_csv('data/maintenance.csv')
    
    # 1. Heat Dissipation Risk
    temp_diff = df['Process temperature [K]'] - df['Air temperature [K]']
    df['heat_risk'] = np.where((temp_diff < 8.6) & (df['Rotational speed [rpm]'] < 1380), 1.0, 0.2)
    
    # 2. Power Failure Risk
    power = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * (2 * np.pi / 60))
    df['power_risk'] = np.where((power < 3500) | (power > 9000), 1.0, 0.1)
    
    # 3. Overstrain Risk
    overstrain = df['Tool wear [min]'] * df['Torque [Nm]']
    df['overstrain_risk'] = overstrain / 11000 
    
    # Final Health Score calculation (0-100)
    df['health_score'] = (1 - df[['heat_risk', 'power_risk', 'overstrain_risk']].max(axis=1)) * 100
    df['health_score'] = df['health_score'].clip(0, 100)
    
    return df[['Product ID', 'health_score', 'heat_risk', 'power_risk', 'overstrain_risk']].tail(10)

# --- LANGCHAIN TOOLS ---
@tool
def check_maintenance_sensors(product_id: str):
    """Retrieves real-time sensor data (temp, speed, torque) for a specific machine ID."""
    df = pd.read_csv('data/maintenance.csv')
    data = df[df['Product ID'] == product_id]
    return data.to_dict() if not data.empty else "Machine ID not found."


@tool
def analyze_sensor_trends(product_id: str, sensor_name: str):
    """Checks if a machine's current sensor reading deviates from the factory average.""" 
    df = pd.read_csv('data/maintenance.csv')
    machine_data = df[df['Product ID'] == product_id]
    
    if machine_data.empty: return "Machine ID not found."
    
    avg_val = df[sensor_name].mean()
    current_val = machine_data[sensor_name].iloc[-1]
    diff_pct = ((current_val - avg_val) / avg_val) * 100
    
    return f"Current {sensor_name}: {current_val}. Factory Average: {avg_val:.2f}. Deviation: {diff_pct:.2f}%"

@tool
def consult_technical_manual(query: str):
    """Searches the WEG Motor Manual (Pinecone) for repair steps and technical guides."""
    from tools.vector_tools import search_manual  # Cleanly imported from the dedicated vector file
    return search_manual(query)

@tool
def run_sql_query(query: str):
    """Execute SQL queries against the 'industrial_ai.db'. Use for complex cross-table analysis."""
    conn = sqlite3.connect('data/industrial_ai.db')
    try:
        result = pd.read_sql_query(query, conn)
        return result.to_string()
    except Exception as e:
        return f"Error in SQL query: {e}"
    finally:
        conn.close()

@tool
def get_market_news(query: str):
    """Fetches real-time supply chain disruptions or price forecasts using Tavily."""
    from tools.search_tools import get_live_market_news 
    return get_live_market_news(query)