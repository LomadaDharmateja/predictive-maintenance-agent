import sys
import os
import pytest
from streamlit.testing.v1 import AppTest

# --- Tell Python to look inside the 'src' folder ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_app_startup():
    """Test if the VULCAN UI loads without crashing."""
    at = AppTest.from_file("src/app.py")
    at.run(timeout=30)
    
    # 1. Ensure no critical crashes happened
    assert not at.exception
    
    # 2. Check if the Sidebar loaded correctly (Streamlit reads this first)
    assert "System Controls" in at.title[0].value

def test_prediction_logic():
    """Test if the ML model still loads and predicts correctly."""
    from tools.data_tools import predict_failure
    
    # LangChain @tool objects require .invoke() with a dictionary of arguments
    test_data = {
        "air_temp": 298.0, 
        "process_temp": 308.0, 
        "speed": 1500.0, 
        "torque": 40.0, 
        "tool_wear": 0.0
    }
    
    result = predict_failure.invoke(test_data)
    
    # Check if the output string contains our expected phrasing
    assert "Failure Probability" in result