import os
from google import genai

# 1. Provide your API key here
API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. Initialize the official client
client = genai.Client(api_key=API_KEY)

print("Fetching models available for content generation...\n")

try:
    # 3. List all models available to your key
    for model in client.models.list():
        # Filter for models that can generate text/responses
        if "generateContent" in model.supported_actions:
            print(f"🔹 Model ID: {model.name}")
            print(f"   Name:     {model.display_name}")
            print(f"   Limits:   Input {model.input_token_limit} tokens | Output {model.output_token_limit} tokens")
            print("-" * 50)
            
except Exception as e:
    print(f"An error occurred. Please verify your API key: {e}")