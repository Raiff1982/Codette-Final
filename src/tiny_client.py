"""
Minimal console client for Codette API
"""
import requests
import sys

def main():
    print("Codette Console Client")
    print("Type 'exit' to quit\n")
    
    # Check server connection
    try:
        response = requests.get("http://127.0.0.1:9000/health")
        if response.status_code == 200:
            print("Connected to Codette server!")
        else:
            print("Server is not responding correctly")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Codette server")
        print("Please make sure the server is running on port 9000")
        sys.exit(1)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            if not question:
                continue
                
            response = requests.post(
                "http://127.0.0.1:9000/ask",
                json={"text": question}
            )
            result = response.json()
            print(f"\nCodette: {result['response']}\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            
    print("\nGoodbye!")

if __name__ == '__main__':
    main()