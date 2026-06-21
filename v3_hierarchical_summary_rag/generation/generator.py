import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import client

def generate_final_response(user_query, retrieved_chunks, conversation_history=""):
    """
    Synthesizes source blocks naturally. 
    If context is present, it forces inline citations.
    If context is missing, it pivots gracefully to provide general concepts 
    while transparently notifying the user.
    """
    
    # Check if we have substantive source material
    context_present = len(retrieved_chunks) > 0
    
    if not context_present:
        context_block = "NO DIRECT MATCHING SOURCE MATERIAL FOUND IN DATABASE."
    else:
        context_segments = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            segment = f"""
                        --- SOURCE BLOCK {idx} ---
                        Document: {chunk['source_document']}
                        Content:
                        {chunk['content']}
                        """
            context_segments.append(segment)
        context_block = "\n".join(context_segments)

    # System prompt enforcing NotebookLM conversational intelligence
    system_instruction = f"""
    You are a highly intuitive, conversational document intelligence engine designed like Google's NotebookLM.
    Your goal is to be maximally helpful, clear, and perfectly honest about where your information comes from.

    CONTEXT STATUS FOR THIS TURN: Context Present = {context_present}

    CORE COGNITIVE DIRECTIVES:

    1. WHEN CONTEXT IS PRESENT:
       - Act as a strict closed-book system for those specific facts.
       - Answer the question directly using the source blocks.
       - Every time you cite a limit, rule, or clause, you MUST append a natural inline citation pointing to the document and page number found inside the source headers (e.g., "[HDFC Policy, Page 12]").
       
    2. WHEN CONTEXT IS ABSENT (THE HYBRID FALLBACK):
       - Do NOT give a robotic error message or refuse to speak.
       - Start your response by transparently notifying the user that your uploaded notebook/database files do not contain the specific answer.
       - Immediately pivot to using your general knowledge to explain the concept conceptually or structurally.
       - Frame this fallback clearly so the user knows this is general advice, not an official rule from their uploaded files.
       - Example style: "I couldn't find a specific rule about that in your uploaded policies. However, in standard corporate guidelines, this usually works by..."

    3. TONAL BOUNDARY:
       - Maintain an elite, conversational, and polished tone.
       - Do not use meta-language like "Based on the source blocks provided". Start directly with the synthesized answer or the transparent boundary disclaimer.
    """

    user_prompt = f"""
    [CONVERSATION HISTORY]
    {conversation_history if conversation_history else "No prior interaction."}

    [RETRIVED SOURCE MATERIAL]
    {context_block}

    [USER QUERY]
    {user_query}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config={'system_instruction': system_instruction}
        )
        return response.text.strip()
        
    except Exception as e:
        return f"❌ ENGINE FAULT: Unable to process response smoothly. Details: {e}"