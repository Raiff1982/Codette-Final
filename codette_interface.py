"""
Unified Codette Interface
Combines web API, Gradio UI, and command-line functionality
"""
import gradio as gr
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import json
import asyncio
from typing import Dict, Any
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from codette import Codette

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodetteInterface:
    def __init__(self):
        """Initialize Codette with configuration"""
        try:
            # Load configuration
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config.json, using defaults: {e}")
            self.config = {
                "host": "127.0.0.1",
                "port": 9000,
                "quantum_fluctuation": 0.07,
                "spiderweb_dim": 5,
                "recursion_depth": 4,
                "perspectives": [
                    "Newton",
                    "DaVinci",
                    "Ethical",
                    "Quantum",
                    "Memory"
                ]
            }

        # Initialize Codette
        try:
            self.codette = Codette(
                user_name="WebUser"
            )
            self.response_memory = []
            self.last_interaction = None
            logger.info("Codette interface initialized")
        except Exception as e:
            logger.error(f"Error initializing interface: {e}")
            raise

    def _is_greeting(self, message: str) -> bool:
        """Check if the message is a greeting"""
        greetings = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "hi codette", "hello codette", "greetings", "howdy", "yo", "hiya", "hey there"
        ]
        return any(g in message.lower() for g in greetings)
    
    def _is_farewell(self, message: str) -> bool:
        """Check if the message is a farewell"""
        farewells = ["bye", "goodbye", "see you", "farewell", "good night",
                    "thanks bye", "bye codette", "goodbye codette"]
        return any(message.lower().startswith(f) for f in farewells)

    def process_message(self, message: str, history: list = None):
        """Process a message through Codette's systems and always return a chat-compatible tuple."""
        try:
            current_time = datetime.now()

            # Ensure history is a list of (user, bot) tuples
            if history is None or not isinstance(history, list):
                history = []

            # Handle rapid repeated messages
            if self.last_interaction and (current_time - self.last_interaction).total_seconds() < 1:
                response = "I'm thinking... Give me just a second! 😊"
                history.append((message, response))
                return "", history

            self.last_interaction = current_time

            # Conversational greeting handling
            if self._is_greeting(message):
                response = (
                    "Hello! 👋 I'm Codette, your AI programming assistant. "
                    "How can I help you today? If you want to talk code, AI, or just chat, I'm here!"
                )
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_response = f"[{timestamp}]\n\n{response}"
                self.response_memory.append({
                    "query": message,
                    "response": response,
                    "timestamp": timestamp
                })
                history.append((message, formatted_response))
                return "", history

            # Process through Codette with error handling

            try:
                result = self.codette.respond(message)
                response = result.get("response", "[No response generated]")
                model_id = result.get("model_id", "unknown")

                # Add metrics to response if available
                if "metrics" in result:
                    metrics = result["metrics"]
                    response += "\n\n📊 Metrics:"
                    if "confidence" in metrics:
                        response += f"\nConfidence: {metrics['confidence']:.1%}"
                    if "quantum_coherence" in metrics:
                        response += f"\nQuantum Coherence: {metrics['quantum_coherence']:.1%}"

                # Add insights if available
                if "insights" in result and result["insights"]:
                    response += "\n\n💡 Insights:\n" + "\n".join(f"• {insight}" for insight in result["insights"])

                # Always show the model in use
                response += f"\n\n🤖 Model in use: {model_id}"

                # If the response is a fallback or unclear, make it more conversational
                fallback_phrases = [
                    "i apologize, but i need to collect my thoughts",
                    "could you please rephrase your question",
                    "i'm not sure i understood",
                    "i'm not sure i understand",
                    "i'm not sure what you mean",
                    "can you rephrase",
                    "please rephrase"
                ]
                response_lower = response.lower()
                if any(phrase in response_lower for phrase in fallback_phrases) or not response.strip():
                    response = (
                        "Hmm, I want to give you the best answer I can! "
                        "Could you clarify or ask in a different way? Or just tell me more about what you're working on. 😊"
                        f"\n\n🤖 Model in use: {model_id}"
                    )

            except Exception as e:
                logger.error(f"Error in response processing: {str(e)}")
                response = f"Sorry, I ran into a hiccup while processing your request: {str(e)}"

            # Format comprehensive response
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_response = f"[{timestamp}]\n\n{response}"

            # Store in memory
            self.response_memory.append({
                "query": message,
                "response": response,
                "timestamp": timestamp
            })

            # Always append to chat history and return
            history.append((message, formatted_response))
            return "", history

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = f"Sorry, I had trouble processing your request. Details: {str(e)}"
            history.append((message, error_msg))
            return "", history

    def clear_history(self):
        """Clear chat history and reset memory"""
        self.response_memory = []
        return [], []

    def search_knowledge(self, query: str) -> str:
        """Search Codette's knowledge base"""
        try:
            result = self.codette.respond(f"Search: {query}")
            return result["response"]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Error performing search: {str(e)}"

    def get_system_state(self) -> str:
        """Get current system state information"""
        try:
            quantum_state = self.codette.quantum_state
            memory_size = len(self.response_memory)
            perspectives = ", ".join(self.codette.perspectives)
            
            return (
                f"🧠 System State:\n"
                f"- Active Perspectives: {perspectives}\n"
                f"- Memory Size: {memory_size} interactions\n"
                f"- Quantum Coherence: {quantum_state.get('coherence', 0.5):.2f}\n"
                f"- Last Update: {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            logger.error(f"Error getting system state: {e}")
            return f"Error retrieving system state: {str(e)}"

    async def run_quantum_simulation(self, cores: int = 4) -> Dict[str, Any]:
        """Run quantum simulation"""
        try:
            return await self.codette.run_quantum_simulation(cores)
        except Exception as e:
            logger.error(f"Quantum simulation error: {e}")
            return {"error": str(e)}

    def create_gradio_interface(self):
        """Create and configure the Gradio interface"""
        with gr.Blocks(title="Codette AI Assistant", theme=gr.themes.Soft()) as app:
            gr.Markdown("""# 🤖 Codette AI Assistant
            Your advanced AI programming assistant with multi-perspective reasoning""")
            
            with gr.Tabs():
                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        bubble_full_width=False,
                        height=450,
                        show_label=False
                    )
                    
                    with gr.Row():
                        txt = gr.Textbox(
                            show_label=False,
                            placeholder="Ask me anything about programming, AI, or technology...",
                            container=False,
                            scale=8
                        )
                        submit_btn = gr.Button("Send", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        state_btn = gr.Button("Show System State")
                    
                    system_state = gr.Markdown()
                    
                with gr.Tab("Knowledge Search"):
                    gr.Markdown("""### 🔍 Knowledge Base Search
                    Search through Codette's extensive knowledge base""")
                    
                    with gr.Row():
                        search_input = gr.Textbox(
                            show_label=False,
                            placeholder="Enter your search query...",
                            container=False
                        )
                        search_btn = gr.Button("Search")
                    
                    search_output = gr.Markdown()
            
            # Set up event handlers
            txt_msg = txt.submit(
                self.process_message,
                [txt, chatbot],
                [txt, chatbot]
            )
            
            btn_msg = submit_btn.click(
                self.process_message,
                [txt, chatbot],
                [txt, chatbot]
            )
            
            clear_btn.click(
                self.clear_history,
                None,
                [chatbot, txt]
            )
            
            state_btn.click(
                self.get_system_state,
                None,
                system_state
            )
            
            search_btn.click(
                self.search_knowledge,
                search_input,
                search_output
            )
            
            search_input.submit(
                self.search_knowledge,
                search_input,
                search_output
            )
        
        return app

    def create_web_app(self):
        """Create Flask web application"""
        app = Flask(__name__)
        app.secret_key = "codette_secret_key_2025"
        CORS(app)
        
        @app.route('/api/query', methods=['POST'])
        def api_query():
            try:
                data = request.get_json()
                query = data.get('query', '')
                if not query:
                    return jsonify({"error": "Query is required"}), 400
                    
                response = self.process_message(query)
                return jsonify(response)
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/health', methods=['GET'])
        def api_health():
            try:
                return jsonify({
                    "status": "healthy",
                    "state": self.get_system_state(),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({"error": str(e)}), 500
                
        @app.route('/api/quantum-simulation', methods=['POST'])
        def api_quantum_simulation():
            try:
                data = request.get_json()
                cores = min(data.get('cores', 4), 16)  # Safety limit
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    results = loop.run_until_complete(
                        self.run_quantum_simulation(cores)
                    )
                    return jsonify(results)
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"Quantum simulation error: {e}")
                return jsonify({"error": str(e)}), 500
        
        return app

def create_interface(interface_type="all"):
    """
    Create Codette interface of specified type
    
    Args:
        interface_type: Type of interface to create ("gradio", "web", or "all")
    """
    interface = CodetteInterface()
    
    if interface_type == "gradio":
        return interface.create_gradio_interface()
    elif interface_type == "web":
        return interface.create_web_app()
    else:
        return interface

if __name__ == "__main__":
    # Create interface instance
    interface = CodetteInterface()
    
    # Create both web and Gradio interfaces
    web_app = interface.create_web_app()
    gradio_app = interface.create_gradio_interface()
    
    # Start web server in a separate thread
    from threading import Thread
    web_thread = Thread(
        target=web_app.run,
        kwargs={
            'host': interface.config.get("host", "127.0.0.1"),
            'port': 5000,
            'debug': False
        }
    )
    web_thread.start()
    
    # Start Gradio interface
    gradio_app.launch(
        server_name=interface.config.get("host", "127.0.0.1"),
        server_port=interface.config.get("port", 9000),
        share=False
    )