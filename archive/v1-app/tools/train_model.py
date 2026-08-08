import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, 'data', 'industrial_ai.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'failure_predictor.pkl')

def train_failure_predictor():
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database not found at {DB_PATH}. Run db_setup.py first!")
        return

    print("🧠 Connecting to Database...")
    conn = sqlite3.connect(DB_PATH)
    
    # Use standard SQL to pull data
    df = pd.read_sql_query("SELECT * FROM maintenance", conn)
    conn.close()

    print(f"📊 Training on {len(df)} rows...")
    # Features (Sensor inputs)
    X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    # Target (What we want to predict)
    y = df['Machine failure']

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Ensure models folder exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save
    joblib.dump(model, MODEL_PATH)
    print(f"✅ ML Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_failure_predictor()