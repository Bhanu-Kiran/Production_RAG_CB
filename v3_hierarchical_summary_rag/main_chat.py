import os
import sys
from pathlib import Path
# Add parent directory to sys.path to allow imports from config
sys.path.append(str(Path(__file__).parent.parent))

from routing.router import analyze_and_route_query
from retrieval.retriever import execute_hybrid_retrieval
from generation.generator import generate_final_response
from config.config import client

def build_history_string(history_list, max_turns=15):
    """Compiles the recent conversational history into a clean string for the LLM."""
    recent_turns = history_list[-max_turns:]
    compiled = []
    for turn in recent_turns:
        compiled.append(f"User: {turn['user']}\nAI: {turn['ai']}")
    return "\n\n".join(compiled)

def main():
    print("\n" + "="*60)
    print(" 📘 WELCOME TO YOUR UPGRADED RAG ENGINE")
    print("      Type your queries below. Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")
    
    conversation_history = []

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("\n👋 Closing Notebook session. Goodbye!")
                break

            print("🤖 Processing query...")
            history_str = build_history_string(conversation_history)
            
            # Step 1: Cognitive Routing
            routing_blueprint = analyze_and_route_query(user_input, conversation_history=history_str)
            intent = routing_blueprint.get("intent", "single_search")
            
            # Step 2: Route Execution
            if intent == "chitchat":
                print("   └─ [Routing: Chitchat Bypass]")
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"Respond naturally to this user greeting/pleasantry, keeping it brief:\n{user_input}"
                )
                final_answer = response.text.strip()
            else:
                search_queries = routing_blueprint.get("search_queries", [])
                targets = routing_blueprint.get("target_documents", [])
                
                print(f"   └─ [Routing: {intent.upper()}] Targets: {targets} | Sub-Queries: {len(search_queries)}")
                
                # Fetch over-fetched, MMR-diversified, and Cross-Encoder reranked chunks
                retrieved_chunks = execute_hybrid_retrieval(
                                                        routing_blueprint, 
                                                        over_fetch_limit=50,   # Stage 1: Pull 50 matching candidates from DB
                                                        mmr_k=20,              # Stage 2: Reduce to 20 unique entries via MMR
                                                        final_k=6              # Stage 3: Return top 6 context items post Cross-Encoding
                                                    )
                
                # Step 3: Synthesis
                final_answer = generate_final_response(
                    user_query=user_input,
                    retrieved_chunks=retrieved_chunks,
                    conversation_history=history_str
                )
            
            print(f"\n🤖 AI:\n{final_answer}")
            print("\n" + "-"*40)
            
            conversation_history.append({"user": user_input, "ai": final_answer})
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ CRITICAL SYSTEM ERROR: {e}")

if __name__ == "__main__":
    main()