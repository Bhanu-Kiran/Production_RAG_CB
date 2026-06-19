import os
import time
import requests
from dotenv import load_dotenv

print("🔍 Loading environment...")
# Force load the .env file
load_dotenv(override=True)

# 1. Manually extract and verify the key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ CRITICAL ERROR: GEMINI_API_KEY is completely blank or missing in your .env file!")
    print("Please check your .env file and ensure it looks exactly like:")
    print("GEMINI_API_KEY=AIzaSyYourKeyHere...")
    exit(1)

# Print a masked version to prove Python can actually see the string
print(f"✅ Key loaded successfully.")
print(f"   ↳ Starts with: {api_key[:6]}...")
print(f"   ↳ Total Length: {len(api_key)} characters")

# 2. Pure REST API Test (Bypassing the Google SDK completely)
print("\n=======================================")
print("🧪 TEST: Pure REST HTTP Request")
print("=======================================")

# Manually construct the Google API URL using your specific key
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

payload = {
    "contents": [{"parts": [{"text": "Reply with exactly two words: 'Connection OK'"}]}]
}
headers = {"Content-Type": "application/json"}

try:
    print("   ↳ Sending raw POST request to Google...")
    start_time = time.time()
    
    # We set a strict 20-second timeout.
    response = requests.post(url, json=payload, headers=headers, timeout=20.0)
    duration = time.time() - start_time
    
    print(f"   ↳ HTTP Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print(f"✅ Success in {duration:.2f} seconds.")
        # Extract the text manually from the JSON response
        answer = response.json()['candidates'][0]['content']['parts'][0]['text']
        print(f"🤖 AI Response: {answer.strip()}")
    else:
        print(f"❌ API Rejected the Request: {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ FAILED: Python's 'requests' library timed out after 20 seconds!")
    print("   ↳ This proves Python in WSL cannot reach the internet, even though 'curl' can.")
except Exception as e:
    print(f"❌ FAILED: {e}")