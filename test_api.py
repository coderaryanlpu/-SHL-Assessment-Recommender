import requests
import json

def test_health():
    print("Testing /health...")
    resp = requests.get("http://localhost:8000/health")
    print(f"Status: {resp.status_code}")
    print(resp.json())
    print()

def test_chat():
    print("Testing /chat...")
    payload = {
        "messages": [
            {"role": "user", "content": "I am hiring a Java developer. Need an assessment."}
        ]
    }
    resp = requests.post("http://localhost:8000/chat", json=payload)
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print("Failed to decode JSON:", resp.text)

if __name__ == "__main__":
    try:
        test_health()
        test_chat()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running? (uvicorn main:app --reload)")
