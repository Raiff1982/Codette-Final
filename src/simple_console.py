"""
Simple console client for Codette API
"""
import requests
import sys
from datetime import datetime

def main():
    print("Codette Console Client")
    print("Type 'exit' to quit\n")
    
    # First check if server is running
    try:
        response = requests.get("http://127.0.0.1:9000/health")
        if response.status_code == 200:
            print("Connected to Codette server successfully!")
        else:
            print("Server is not responding correctly")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Codette server")
        print("Please make sure the server is running on port 9000")
        sys.exit(1)
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nYour question: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Send request to server
            response = requests.post(
                "http://127.0.0.1:9000/ask",
                json={"text": user_input},
                timeout=30
            )
            
            # Handle response
            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    print(f"\nCodette: {data['response']}")
                else:
                    print("\nError: Unexpected response format")
            else:
                print(f"\nError: Server returned status code {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("\nError: Lost connection to Codette server")
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == '__main__':
    main()