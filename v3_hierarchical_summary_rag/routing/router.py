import json
import sys
from pathlib import Path

# Add parent directory to sys.path to allow imports from config
sys.path.append(str(Path(__file__).parent.parent))

from config.config import client

def analyze_and_route_query(user_query, conversation_history="", use_case="Health Insurance Bot"):
    """
    Acts as the cognitive routing layer. 
    Analyzes the user's prompt and decides exactly how the database should be queried.
    """
    
    prompt = f"""
    You are an elite AI query routing agent for an advanced {use_case} Bot utilizing RAG.
    Analyze the incoming User Query and the provided Conversation History.

    [CONVERSATION HISTORY]
    {conversation_history}

    [USER QUERY]
    {user_query}

    Your job is to map this input into a structured JSON configuration block.
    
    Determine the following:
    1. intent: Choose exactly one of: 
       - "chitchat" (pleasantries, greetings, thank yous - no database search needed)
       - "single_search" (asking about a concept, rule, or extracting info from a specific doc or generally)
       - "comparison" (explicitly comparing rules or concepts across MULTIPLE specific documents)
    2. target_documents: Extract an array of document names/brands explicitly mentioned (e.g., ["HDFC", "SBI", "Tata"]). If general or none, return an empty array [].
    3. search_queries: An array of highly optimized, context-fused search strings.
       * CRITICAL: Each query must be a comparably mid-sized targeted phrase.
       * Strip out conversational filler (e.g., do NOT generate "Can you tell me what the rules are for...").
       * Turn complex paragraphs into separate, clean atomic lookups.
    ---
    OUTPUT FIELDS DETERMINATION:
    
    - "intent": Must be exactly one of: "chitchat", "single_search", or "comparison".
    - "target_documents": An array of base brand names ONLY. Strip all symbols like '+', '-', or 'etc'. (e.g., "HDFC+" -> "HDFC"). If global, leave empty: [].
    - "search_queries": An array of standalone, perfectly spelled search strings. 
       * CRITICAL LIMIT: You MUST NOT generate more than 3 search queries total, no matter how complex the prompt is.
       * Keep queries short (5-10 words).

    Output ONLY a valid, minified JSON object with keys: "intent", "target_documents", "search_queries". Do not include markdown code blocks.
    """
    
    try:
        # Utilizing the modern SDK pattern matching your config
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        routing_payload = json.loads(clean_text)
        return routing_payload
        
    except Exception as e:
        print(f"      [!] Routing Engine Exception: {e}. Falling back to default global search.")
        return {
            "intent": "single_search",
            "target_documents": [],
            "search_queries": [user_query]
        }