# generator.py
from config.config import client

def generate_response(user_query, context_chunks):
    """
    Assembles extracted text assets and generates an objective answer
    with exact semantic citation tags.
    """
    # Compile text context blocks with explicit source mapping tags
    compiled_context = ""
    if context_chunks:
        for idx, chunk in enumerate(context_chunks, 1):
            compiled_context += f"\n--- Context Source [{idx}]: {chunk['source_policy']} ---\n"
            compiled_context += f"{chunk['content']}\n"
    else:
        compiled_context = "No specific document sections found matching the query."

    system_prompt = """
    You are an elite, objective healthcare insurance copilot. Your task is to resolve user policy calculations and definitions using ONLY the provided verified context fragments.

    Execution Protocols:
    1. Base all assertions strictly on the context provided.
    2. If explicit policy context data is missing, state clearly that the answer could not be located in current records.
    3. Cite your sources cleanly by referencing the specific 'Context Source' title provided.
    """

    user_payload = f"""
    [CONTEXT DATA PORTAL]
    {compiled_context}

    [USER QUESTION]
    {user_query}
    
    Provide a clear, insight-driven response matching the tone and constraints above.
    """

    try:
        response = client.models.generate_content(
            model='gemini-3-flash', # Premium tier generation model balancing intelligence and speed
            contents=user_payload,
            config={"system_instruction": system_prompt}
        )
        return response.text
    except Exception as e:
        return f"❌ Generation layer timed out or crashed: {e}"
