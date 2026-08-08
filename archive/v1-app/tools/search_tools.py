import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Initialize Tavily
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def get_live_market_news(query):
    """Searches the web for real-time industrial and commodity news."""
    print(f"🌐 Searching the web for: {query}...")
    # search_depth="advanced" gives you a deeper analysis
    response = tavily.search(query=query, search_depth="advanced", max_results=5)
    print(f"DEBUG: Found {len(response['results'])} news articles.")
    
    # Format the results for our LLM
    context = ""
    for result in response['results']:
        context += f"\nSource: {result['url']}\nContent: {result['content']}\n"
    
    return context