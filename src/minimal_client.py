"""
Simple command-line client for Codette using the unified interface
"""

import sys
sys.path.append('..')  # Add parent directory to path

from codette_interface import create_interface

def main():
    print("Codette Test Client")
    print("Type 'exit' to quit")
    
    # Create interface instance
    interface = create_interface()
    
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
            
            # Process message
            result = interface.process_message(user_input)
            if "error" in result:
                print(f"\nError: {result['error']}")
            else:
                print(f"\nCodette: {result['response']}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)