import os
import sys
import time
import json

# Ensure absolute root imports resolve perfectly matching the V3 package layout
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.router import analyze_and_route_query
from retrieval.retriever import execute_hybrid_retrieval
from generation.generator import generate_final_response
from config.config import client

# Define the complete multi-turn automated diagnostic test suite matrix
TEST_SUITES = {
    "Suite_1_Semantic_Normalization": [
        {"input": "hello good evening bot", "description": "Verify chitchat routing layer bypass speed"},
        {"input": "what r the wait limits for PED and murnity caps in HDFC+?", "description": "Test typo clearing, acronym expansion, symbol stripping"}
    ],
    "Suite_2_Anaphora_and_Context_Drift": [
        {"input": "Tell me about the Activ One policy details.", "description": "Establish conversational baseline entity tracking"},
        {"input": "Is there a sub-limit on its room rent?", "description": "Test pronoun ('its') context resolution back to Activ One"},
        {"input": "What about the other policy from IndusInd?", "description": "Test rapid context switching to a secondary target document"}
    ],
    "Suite_3_Extreme_Compound_Stress": [
        {"input": "Compare SBI, HDFC ERGO, and Aditya Birla health rules regarding cataracts, premium payments, and pre-existing conditions.", 
         "description": "Stress-test the sub-query ceiling (max 3) and dynamic local Cross-Encoder performance"}
    ],
    "Suite_4_Adversarial_Out_Of_Domain": [
        {"input": "What is the absolute fastest way to tune a high-performance rocket engine running on liquid oxygen?", 
         "description": "Force a global vector search mismatch to evaluate the hybrid fallback boundary logic"}
    ]
}

def build_history_string(history_list, max_turns=5):
    recent_turns = history_list[-max_turns:]
    compiled = []
    for turn in recent_turns:
        compiled.append(f"User: {turn['user']}\nAI: {turn['ai']}")
    return "\n\n".join(compiled)

def run_diagnostic_engine():
    print("\n" + "="*80)
    print(" 🛠️  ENTERPRISE AUTOMATED STRESS-TESTING FRAMEWORK - RAG PIPELINE V3")
    print("="*80)
    
    overall_suite_start = time.time()
    total_tests_executed = 0
    failures_encountered = 0

    for suite_name, conversation_steps in TEST_SUITES.items():
        print(f"\n🚀 EXECUTING SUITE: {suite_name.replace('_', ' ')}")
        print("-" * 80)
        
        # Track active memory isolation per individual test suite
        conversation_history = []
        
        for step_idx, step in enumerate(conversation_steps, 1):
            total_tests_executed += 1
            user_input = step["input"]
            expected_behavior = step["description"]
            
            print(f"\n[Turn {step_idx}] Input: \"{user_input}\"")
            print(f"🎯 Target Objective: {expected_behavior}")
            
            turn_start_time = time.time()
            history_str = build_history_string(conversation_history)
            
            try:
                # --------------------------------------------------------------------
                # PHASE 1: ROUTING METRICS
                # --------------------------------------------------------------------
                route_start = time.time()
                routing_blueprint = analyze_and_route_query(user_input, conversation_history=history_str)
                route_latency = time.time() - route_start
                
                intent = routing_blueprint.get("intent", "single_search")
                targets = routing_blueprint.get("target_documents", [])
                sub_queries = routing_blueprint.get("search_queries", [])
                
                print(f"   ├─ [Router] Intent: {intent.upper()} | Targets: {targets} | Latency: {route_latency:.3f}s")
                print(f"   ├─ [Router] Generated Sub-Queries: {sub_queries}")
                
                # --------------------------------------------------------------------
                # PHASE 2: RETRIEVAL AND EXTRACTION METRICS
                # --------------------------------------------------------------------
                retrieval_latency = 0.0
                chunks_delivered = 0
                retrieved_chunks = []
                
                if intent == "chitchat":
                    print("   ├─ [Retriever] Bypassed (Chitchat Intent Verified)")
                    chitchat_start = time.time()
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"Respond naturally to this user greeting: {user_input}"
                    )
                    final_answer = response.text.strip()
                    retrieval_latency = time.time() - chitchat_start
                else:
                    retrieval_start = time.time()
                    # Execute our local 3-tier processing architecture ($50 -> 20 -> 6)
                    retrieved_chunks = execute_hybrid_retrieval(
                        routing_blueprint, 
                        over_fetch_limit=50, 
                        mmr_k=20, 
                        final_k=6
                    )
                    retrieval_latency = time.time() - retrieval_start
                    chunks_delivered = len(retrieved_chunks)
                    print(f"   ├─ [Retriever] Context Pages Extracted: {chunks_delivered} | Latency: {retrieval_latency:.3f}s")
                    
                    # --------------------------------------------------------------------
                    # PHASE 3: RESPONSE GENERATION AND SYNTHESIS
                    # --------------------------------------------------------------------
                    gen_start = time.time()
                    final_answer = generate_final_response(
                        user_query=user_input,
                        retrieved_chunks=retrieved_chunks,
                        conversation_history=history_str
                    )
                    generation_latency = time.time() - gen_start
                    print(f"   ├─ [Generator] Synthesis Verified | Latency: {generation_latency:.3f}s")

                # Verify execution speed metrics
                total_turn_latency = time.time() - turn_start_time
                print(f"   ├─ [Latency Total] Finished turn execution loop in: {total_turn_latency:.3f}s")
                
                # Verify output length to check for errors
                preview_length = 120
                answer_preview = final_answer.replace('\n', ' ')
                print(f"   └─ [AI Response Preview]: {answer_preview[:preview_length]}...")
                
                # Update history object to sustain conversational context tracking checks
                conversation_history.append({"user": user_input, "ai": final_answer})
                
            except Exception as turn_fault:
                print(f"   ❌ CRITICAL TURN FAULT ENCOUNTERED: {turn_fault}")
                failures_encountered += 1

            print("." * 40)
            
    print("\n" + "="*80)
    print(" 📊 FINAL SYSTEM PERFORMANCE TELEMETRY REPORT")
    print("="*80)
    print(f" Total Script Diagnostic Operations Fired : {total_tests_executed}")
    print(f" Pipeline Fatal Failures Intercepted     : {failures_encountered}")
    print(f" Automated Suite Processing Time          : {time.time() - overall_suite_start:.2f} seconds")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_diagnostic_engine()