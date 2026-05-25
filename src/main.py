import os
import time  # 1. Added time module for rate-limiting protection
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory

from tools.data_tools import (
    analyze_sensor_trends, 
    check_maintenance_sensors, 
    consult_technical_manual, 
    run_sql_query, 
    get_market_news, 
    predict_failure,
    get_failed_machines,           
    get_supplier_info,             
    get_internal_commodity_prices  
)

load_dotenv()

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/factory_brain.log',
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("VULCAN")

class IndustrialAI:
    def __init__(self):
        logger.info("VULCAN AI Brain initializing with Gemini 2.5 Flash-Lite.")
        
        # Use the latest Flash-Lite for high-speed, low-cost inference
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", 
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3 
        )

        # Tools provided to the Agent
        self.tools = [
            analyze_sensor_trends, 
            check_maintenance_sensors, 
            consult_technical_manual, 
            run_sql_query, 
            get_market_news,                
            predict_failure,
            get_failed_machines,            
            get_supplier_info,              
            get_internal_commodity_prices   
        ]
        
        # Unified System Persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Industrial Systems Engineer at VULCAN OS. 
            You have autonomous access to sensors, manuals, and market data.
            Rules:
            1. If a sensor is abnormal, check the technical manual using 'consult_technical_manual'.
            2. If a part needs replacement, check 'get_internal_commodity_prices' or 'get_supplier_info'. 
               (Note: Ensure you use the exact tool names available to you).
            3. Always provide a factual, engineering-based reasoning."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, memory=self.memory, verbose=True)

    def run_analysis(self):
        """Runs a fully autonomous executive system audit with rate-limit handling."""
        prompt = """
        FACTORY AUDIT TASK:
        1. Use 'get_failed_machines' to identify how many units have failed and their codes.
        2. Pick the most critical failure (e.g., PWF or HDF).
        3. Research the repair steps in the technical manual.
        4. Check the market for part prices.
        
        Provide a summary of the total failures found and a deep-dive plan for the most urgent one.
        """
        
        # 2. Wrap the execution loop so your UI app.py doesn't completely crash on a 429
        try:
            logger.info("Starting factory audit execution...")
            response = self.executor.invoke({"input": prompt})
            
            # 3. Rate-Limit Buffer: Give the Free Tier API key 4-5 seconds to breathe 
            # before app.py allows another audit request to fire.
            time.sleep(1)
            return response["output"]
            
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("⚠️ Free Tier limit exhausted! Initiating cooldown delay...")
                time.sleep(15) # Force wait to reset window
                return "The system is cooling down from an API rate-limiting block. Please try again in a moment."
            else:
                return f"An error occurred: {str(e)}"