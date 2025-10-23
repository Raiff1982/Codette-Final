"""
Interactive client for Codette AI
"""

import asyncio
import json
import sys
from datetime import datetime
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

console = Console()

class CodetteClient:
    def __init__(self, base_url="http://127.0.0.1:9000"):
        self.base_url = base_url
        self.session = requests.Session()

    def check_server(self):
        """Check if Codette server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

    def send_query(self, query: str):
        """Send a query to Codette (via /ask endpoint)"""
        try:
            data = {"text": query}
            response = self.session.post(f"{self.base_url}/ask", json=data)
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_system_state(self):
        """Get current system state (via /health endpoint)"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

def display_response(response):
    """Display Codette's response with rich formatting"""
    try:
        if response.get("status") == "error":
            console.print(f"[red]Error: {response.get('error')}[/red]")
            return

        # Display main response
        if "response" in response:
            console.print(Panel(
                Markdown(response["response"]),
                title="Codette's Response",
                border_style="blue"
            ))

        # Display additional insights
        if "insights" in response:
            console.print(Panel(
                str(response["insights"]),
                title="Insights",
                border_style="magenta"
            ))

        # Display metrics if available
        if "metrics" in response:
            console.print(Panel(
                json.dumps(response["metrics"], indent=2),
                title="Metrics",
                border_style="green"
            ))

        # Display health status
        if "health" in response:
            console.print(Panel(
                json.dumps(response["health"], indent=2),
                title="Health Status",
                border_style="cyan"
            ))

    except Exception as e:
        console.print(f"[red]Error displaying response: {e}[/red]")

def main():
    console.clear()
    console.print("[bold blue]Welcome to Codette Interactive Client[/bold blue]")
    console.print("Type 'exit' to quit, 'help' for commands\n")
    
    client = CodetteClient()

    # Check server connection
    if not client.check_server():
        console.print("[red]Error: Cannot connect to Codette server. Make sure it's running on http://127.0.0.1:9000[/red]")
        return

    console.print("[green]Connected to Codette server[/green]")

    while True:
        try:
            command = Prompt.ask("\n[bold cyan]Enter your query[/bold cyan]")

            if command.lower() == 'exit':
                break
            elif command.lower() == 'help':
                console.print(Panel("""
                    [bold]Available Commands:[/bold]
                    - Any text: Send as query to Codette
                    - status: Show system state
                    - help: Show this help
                    - exit: Quit the client
                """, title="Help"))
                continue
            elif command.lower() == 'status':
                state = client.get_system_state()
                console.print(Panel(json.dumps(state, indent=2), title="System State"))
                continue

            # Send query and display response
            response = client.send_query(command)
            display_response(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[bold blue]Goodbye![/bold blue]")

if __name__ == "__main__":
    main()