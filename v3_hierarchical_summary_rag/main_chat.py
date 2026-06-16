# main_chat.py
import sys
from routing.router import analyze_and_route_query
from retrieval.retriever import execute_hybrid_search
from generation.generator import generate_response

def main():
    print("===============================================================")
    print(" 🏥 HEALTHCARE INSURANCE CO-PILOT (MODULAR PRODUCTION RAG)     ")
    print("===============================================================")
    
    # Simple operational conversational history tracking array
    history = []

    while True:
        try:
            user_input = input("\nAsk a question about the policies (or type 'exit'): ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down co-pilot link. Goodbye.")
                sys.exit(0)
                
            if not user_input:
                continue

            # Compile standard string representation of recent history
            history_str = "\n".join(history[-4:])
            
            # STAGE 1: INTENT & METADATA ROUTING
            print("\n⚡ [1/4] Routing query and determining constraints...")
            routing = analyze_and_route_query(user_input, history_str)
            
            intent = routing.get("intent", "single_search")
            companies = routing.get("target_companies", [])
            sub_queries = routing.get("search_queries", [user_input])
            
            print(f"   └── [Routing Profile]: Intent='{intent}', Entities={companies}")

            # STAGE 2: CONTROL LAYER INTERCEPTION (CHITCHAT SAFETY VALVE)
            if intent == "chitchat":
                print("💬 [2/4] Chitchat detected. Bypassing document stores...")
                # Handle chitchat natively via rapid generation pass without DB hit
                from config.config import client
                response = client.models.generate_content(model='gemini-3.1-flash-lite', contents=user_input)
                print(f"\n========================= AI RESPONSE =========================\n{response.text}")
                print("===============================================================")
                continue

            # STAGE 3: DATA GATHERING & DECOMPOSITION RESOLUTION
            print("🔍 [2/4] Executing isolated database hybrid searches...")
            aggregated_chunks = []
            
            # Loop through sub-queries generated during decomposition
            for sub_q in sub_queries:
                print(f"   ├── Searching vector index for: '{sub_q}'")
                # Retrieve top chunks for each individual sub-query to prevent starvation
                chunks = execute_hybrid_search(sub_q, companies, limit=4)
                aggregated_chunks.extend(chunks)

            # Deduplicate items by text value if identical fragments bleed into cross-search windows
            unique_chunks = {c['content']: c for c in aggregated_chunks}.values()
            
            # STAGE 4: RESPONSE SYNTHESIS
            print("📝 [3/4] Fusing balanced context blocks...")
            print("💬 [4/4] Finalizing text output sequence...")
            final_answer = generate_response(user_input, list(unique_chunks))
            
            print(f"\n========================= AI RESPONSE =========================\n{final_answer}")
            print("===============================================================")
            
            # Append exchange state to memory array
            history.append(f"User: {user_input}")
            history.append(f"AI: {final_answer}")

        except KeyboardInterrupt:
            print("\nLink interrupted. Exiting session.")
            break

if __name__ == "__main__":
    main()
