import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory

# Import your clean tools
from tools.data_tools import (
    analyze_sensor_trends, 
    check_maintenance_sensors, 
    check_market_prices, 
    consult_technical_manual, 
    run_sql_query, 
    get_market_news, 
    predict_failure,
    get_failed_machines
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
            model="gemini-2.5-flash-lite", 
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3 # Lowered for more factual industrial responses
        )

        # Tools provided to the Agent
        self.tools = [
            analyze_sensor_trends, 
            check_maintenance_sensors, 
            check_market_prices, 
            consult_technical_manual, 
            run_sql_query, 
            get_market_news, 
            predict_failure
        ]

        # Unified System Persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Industrial Systems Engineer at VULCAN OS. 
            You have autonomous access to sensors, manuals, and market data.
            Rules:
            1. If a sensor is abnormal, check the technical manual using 'consult_technical_manual'.
            2. If a part needs replacement, check 'check_market_prices'.
            3. Always provide a factual, engineering-based reasoning."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, memory=self.memory, verbose=True)

    def run_analysis(self):
        """Runs the high-level executive report."""
        failures = get_failed_machines()
        if not failures:
            return "All systems nominal. No critical failures detected in current logs."

        target = failures[0]
        # We ask the agent to perform a multi-step investigation autonomously
        prompt = f"Perform a full root-cause analysis for Machine {target['product_id']} experiencing {target['failure_type']}. Check the manual for fix steps and check the market for spare part pricing."
        
        response = self.executor.invoke({"input": prompt})
        return response["output"]