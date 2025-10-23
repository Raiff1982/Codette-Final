"""
Streamlined console client for Codette
"""

import cmd
import requests
import sys
import json

class CodetteConsole(cmd.Cmd):
    intro = 'Welcome to the Codette console. Type help or ? to list commands.\n'
    prompt = 'Codette> '
    
    def do_ask(self, arg):
        """Ask Codette a question: ask What is quantum computing?"""
        if not arg:
            print("Please provide a question")
            return
            
        try:
            response = requests.post(
                "http://localhost:8080/ask",
                json={"text": arg},
                timeout=30
            )
            result = response.json()
            if "response" in result:
                print(f"\nCodette: {result['response']}\n")
            elif "error" in result:
                print(f"\nError: {result['error']}\n")
            else:
                print("\nNo response received\n")
        except Exception as e:
            print(f"\nError: Could not reach Codette server - {str(e)}\n")

    def do_exit(self, arg):
        """Exit the console"""
        print("\nGoodbye!")
        return True
        
    def do_EOF(self, arg):
        """Exit on Ctrl-D (EOF)"""
        print("\nGoodbye!")
        return True
        
    # Shortcuts
    do_quit = do_exit
    do_q = do_exit

if __name__ == '__main__':
    try:
        CodetteConsole().cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)