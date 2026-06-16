import os
import sys
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

# Ensure we can load environment variables
load_dotenv('/home/bhanu/prod-rag-pipeline/.env')

# Initialize local Gemini client for test runner diagnostics
client = genai.Client()

# Import only what is available in the user's unmodified chat_V2.py
try:
    from chat_V2 import (
        get_google_embedding, 
        retrieve_hybrid_context,
        CHAT_HISTORY,
        condense_query
    )
except ImportError as e:
    print(f"❌ Error importing from chat_V2: {e}")
    sys.exit(1)

def assemble_prompt(user_question, results):
    """Replicates the prompt assembly logic from chat_V2 for diagnostic display."""
    context_blocks = []
    for row in results:
        file_name, content, score = row
        context_blocks.append(f"--- SOURCE DOCUMENT: {file_name} (Score: {score:.3f}) ---\n{content}\n")
    full_context = "\n".join(context_blocks)

    generation_prompt = f"""
    You are an expert, highly accurate healthcare insurance assistant. 
    Answer the user's question using ONLY the provided policy context below.
    If the answer is not contained in the context, say "I cannot find the answer in the provided documents."
    Always cite the Source Document name when providing facts. Do not hallucinate.
    
    USER QUESTION: {user_question}
    
    CONTEXT:
    {full_context}
    """
    return generation_prompt

def generate_answer(prompt):
    """Replicates the Gemini generation call logic from chat_V2 for diagnostics."""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"   ❌ [Error] Text generation failed: {e}")
        return None

def run_scenario(scenario_name, query, limit=10):
    """
    Orchestrates and prints all stages of the RAG pipeline for a given query:
    1. Query embedding extraction
    2. Hybrid database search (RRF) and chunk analysis
    3. LLM Prompt compilation
    4. Text generation via Gemini
    """
    print("\n" + "="*90)
    print(f"🚀 SCENARIO: {scenario_name}")
    print(f"❓ QUERY: \"{query}\"")
    print(f"📊 LIMIT: {limit} chunks")
    print("="*90)
    
    # 1. VECTORIZE USER QUERY
    print("\n🧠 [Step 1] Extracting semantic vector via gemini-embedding-2...")
    query_vector = get_google_embedding(query)
    if not query_vector:
        print("   ❌ Failed to calculate embedding vector.")
        return
    print("   ✓ Vector calculated successfully.")
    
    # 2. RETRIEVE CHUNKS
    print("\n🔍 [Step 2] Executing hybrid pgvector + full-text search...")
    results = retrieve_hybrid_context(query, query_vector, limit=limit)
    if not results:
        print("   ⚠️ No relevant document elements retrieved from database.")
        return
        
    print(f"   ✓ Retrieved {len(results)} chunks from database. Sources breakdown:")
    for idx, row in enumerate(results, 1):
        file_name, content, score = row
        # Clean page boundaries for preview
        snippet = " ".join(content.replace("\n", " ").split())
        preview = snippet[:120] + "..." if len(snippet) > 120 else snippet
        print(f"     [{idx}] File: {file_name:<35} | RRF Score: {score:.5f}")
        print(f"         Snippet: \"{preview}\"")
        
    # 3. ASSEMBLE PROMPT
    print("\n📝 [Step 3] Compiling prompt with negative-constraint instructions...")
    prompt = assemble_prompt(query, results)
    
    # Print the start of the prompt for visual verification
    prompt_lines = prompt.strip().split('\n')
    print("   --- PROMPT PREVIEW ---")
    for line in prompt_lines[:15]:
        print(f"   | {line}")
    print("   | ... [CONTEXT BLOCKS OMITTED FOR PREVIEW] ...")
    
    # 4. RUN LLM GENERATION
    print("\n💬 [Step 4] Requesting response from gemini-2.5-flash...")
    response_text = generate_answer(prompt)
    
    print("\n============================= AI RESPONSE =============================")
    if response_text:
        print(response_text)
    else:
        print("   ❌ Failed to generate response.")
    print("=======================================================================")

def run_stateful_scenario(scenario_name, queries, limit=10):
    """
    Simulates a multi-turn conversation to verify:
    1. Chat History tracking
    2. Query Condensation (context-aware rewriting)
    3. Conversational RAG pipeline grounding
    """
    print("\n" + "="*90)
    print(f"🚀 STATEFUL SCENARIO: {scenario_name}")
    print("="*90)
    
    # Clear any previous run history
    CHAT_HISTORY.clear()
    
    for turn, query in enumerate(queries, 1):
        print(f"\n💬 --- Turn {turn}: User asks \"{query}\" ---")
        
        # 1. CONDENSE QUERY
        print("⚡ [1/5] Analyzing conversational context...")
        condensed_query_text = condense_query(query, CHAT_HISTORY)
        if condensed_query_text != query:
            print(f"   🔄 [Optimized Query]: \"{condensed_query_text}\"")
        else:
            print("   ✓ Query is independent.")
            
        # 2. VECTORIZE CONDENSED QUERY
        print("🧠 [2/5] Embedding targeted search keys...")
        query_vector = get_google_embedding(condensed_query_text)
        if not query_vector:
            print("   ❌ Failed to calculate embedding vector.")
            return
            
        # 3. RETRIEVE CONTEXT
        print("🔍 [3/5] Executing PostgreSQL Hybrid Search (Dense + Sparse)...")
        results = retrieve_hybrid_context(condensed_query_text, query_vector, limit=limit)
        if not results:
            print("   ⚠️ No matching policies found in the database.")
            return
            
        print(f"   ✓ Retrieved {len(results)} chunks. Sources breakdown:")
        for idx, row in enumerate(results, 1):
            file_name, content, score = row
            snippet = " ".join(content.replace("\n", " ").split())[:80] + "..."
            print(f"     [{idx}] {file_name:<35} (Score: {score:.5f}) | Snippet: {snippet}")
            
        # 4. ASSEMBLE PROMPT
        print("📝 [4/5] Syncing database blocks into generation prompt...")
        prompt = assemble_prompt(query, results)
        
        # 5. GENERATE ANSWER
        print("💬 [5/5] Requesting comprehensive answer from Gemini...")
        response_text = generate_answer(prompt)
        
        print("\n============================= AI RESPONSE =============================")
        if response_text:
            print(response_text)
            CHAT_HISTORY.append((query, response_text))
        else:
            print("   ❌ Failed to generate response.")
        print("=======================================================================")
        
        if turn < len(queries):
            input("\nPress Enter to proceed to the next turn in the conversation...")

def show_menu():
    scenarios = {
        "1": ("Standard ICU Pricing Lookup", "What is the ICU category room limit in SBI policy?", 15),
        "2": ("Punctuation & Syntax Symbols", "Does the policy cover room/rent? Or is it excluded & restricted?", 15),
        "3": ("Out of Domain (Negative Constraints)", "Does this cover space travel injuries or cosmic radiation sickness?", 15),
        "4": ("Multi-Policy Comparison", "Compare the room rent limits between Aditya Health Insurance and ReAssure 3.0", 15),
        "5": ("Detailed Co-payment Table", "What are the co-payment structures under different variants of ReAssure 3.0?", 15),
        "6": ("Stateful Conversational Follow-up", ["What is the room rent limit for the SBI policy?", "What if I choose a twin sharing room instead?"], 15),
        "7": ("Postgres Stopwords & SQL Safety", "and or not select where & | twin sharing limit", 15)
    }
    
    while True:
        print("\n" + "#"*55)
        print("   HEALTHCARE INSURANCE RAG - SCENARIO ANALYSIS")
        print("#"*55)
        for key, (name, query, limit) in scenarios.items():
            query_desc = f"\"{query}\"" if isinstance(query, str) else f"{query}"
            print(f"  {key}. {name}")
            print(f"     Query/Turns: {query_desc}")
        print("  8. Run ALL scenarios sequentially")
        print("  0. Exit")
        print("#"*55)
        
        choice = input("\nSelect a scenario to run and analyze (0-8): ").strip()
        
        if choice == "0":
            print("Exiting test script. Goodbye!")
            break
        elif choice == "8":
            for key, (name, query, limit) in scenarios.items():
                if isinstance(query, list):
                    run_stateful_scenario(name, query, limit)
                else:
                    run_scenario(name, query, limit)
                input("\nPress Enter to proceed to the next scenario...")
        elif choice in scenarios:
            name, query, limit = scenarios[choice]
            if isinstance(query, list):
                run_stateful_scenario(name, query, limit)
            else:
                run_scenario(name, query, limit)
        else:
            print(f"❌ Invalid selection. Please enter a number between 0 and 8.")

if __name__ == "__main__":
    show_menu()
