import pandas as pd
import sqlite3
import numpy as np
import os

# This finds the absolute path to your project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, 'data', 'industrial_ai.db')
MAINT_CSV = os.path.join(BASE_DIR, 'data', 'maintenance.csv')

def migrate_to_sql():
    if not os.path.exists(MAINT_CSV):
        print(f"❌ Error: Could not find {MAINT_CSV}")
        return

    # 1. Load data
    maint_df = pd.read_csv(MAINT_CSV)
    
    # 2. "Inflate" the data (Generate 50,000 synthetic rows for a 'Huge' feel)
    print("🎈 Inflating dataset to 50,000 rows for robustness...")
    large_maint = pd.concat([maint_df] * 5, ignore_index=True) # Multiply by 5
    # Add some random noise so they aren't exact copies
    large_maint['Torque [Nm]'] = large_maint['Torque [Nm]'] + np.random.normal(0, 2, large_maint.shape[0])
    
    # 3. Connect to SQLite (creates industrial_ai.db)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 4. Save to SQL
    large_maint.to_sql('maintenance', conn, if_exists='replace', index=False)
    pd.read_csv('data/commodity.csv').to_sql('commodities', conn, if_exists='replace', index=False)
    pd.read_csv('data/industrial_cleaned.csv').to_sql('logistics', conn, if_exists='replace', index=False)
    
    conn.close()
    print("✅ Database created: data/industrial_ai.db")

if __name__ == "__main__":
    migrate_to_sql()