# router.py
import json
from config.config import client

def analyze_and_route_query(user_query, conversation_history=""):
    """
    Evaluates conversational context and extracts intent, sub-queries, 
    and target metadata filters using Gemini 3.1 Flash Lite.
    """
    prompt = f"""
    You are an advanced query routing and optimization agent for a healthcare insurance RAG pipeline.
    Analyze the incoming User Query and the provided Conversation History.

    [CONVERSATION HISTORY]
    {conversation_history}

    [USER QUERY]
    {user_query}

    Your job is to decompose and map this input into a structured JSON configuration block.
    
    Determine the following:
    1. intent: Choose exactly one of: 
       - "chitchat" (pleasantries, greetings, thank yous)
       - "single_search" (asking about a rule, definition, or limit for a single policy or general info)
       - "comparison" (explicitly comparing or checking rules across multiple companies/policies)
    2. target_companies: Extract an array of company/policy names explicitly mentioned (e.g., ["HDFC", "SBI"]). If general or none, return an empty array [].
    3. search_queries: An array of optimized, standalone search strings. 
       - If intent is "single_search", provide exactly 1 optimized search query rewritten to remove pronouns.
       - If intent is "comparison", DECOMPOSE the question into separate, standalone search queries for EACH target company to prevent database starvation (e.g., ["HDFC aggregate deductible rules", "SBI aggregate deductible rules"]).
       - If intent is "chitchat", return an empty array [].

    Output ONLY a valid, minified JSON object with keys: "intent", "target_companies", "search_queries". Do not wrap in markdown code blocks.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite', # Ultra-fast performance layer optimized for structural routing
            contents=prompt
        )
        
        # Clean potential markdown formatting
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        routing_payload = json.loads(clean_text)
        return routing_payload
        
    except Exception as e:
        print(f"      [!] Routing Engine Exception: {e}. Falling back to default baseline search.")
        return {
            "intent": "single_search",
            "target_companies": [],
            "search_queries": [user_query]
        }
