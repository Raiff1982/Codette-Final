"""
Bare minimum client
"""
import requests

while True:
    text = input("Question (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break
        
    try:
        r = requests.post("http://127.0.0.1:9000/ask", json={"text": text})
        print(r.json()["response"])
    except Exception as e:
        print(f"Error: {e}")