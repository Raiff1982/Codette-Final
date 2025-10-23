"""
Minimal test client for Codette
"""

import requests
import sys

def main():
    print("Codette Test Client")
    print("Type 'exit' to quit")
    
    while True:
        try:
            question = input("\nYour question: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            response = requests.post(
                "http://localhost:8080/ask",
                json={"text": question},
                timeout=30
            )
            result = response.json()
            
            if "response" in result:
                print(f"\nCodette: {result['response']}")
            elif "error" in result:
                print(f"\nError: {result['error']}")
            else:
                print("\nNo response received")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: Could not reach Codette server - {str(e)}")
            
    print("\nGoodbye!")

if __name__ == '__main__':
    main()