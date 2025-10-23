"""
Simple Codette client
"""

import requests
from typing import Optional, Dict, Any

def ask_codette(question: str) -> Optional[str]:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"text": question}
        )
        data = response.json()
        return data.get("response", data.get("error", "No response"))
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Simple Codette Client")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            question = input("\nYou: ")
            if question.lower() == "exit":
                break
                
            if question.strip():
                response = ask_codette(question)
                print(f"\nCodette: {response}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()