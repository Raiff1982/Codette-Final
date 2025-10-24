#!/usr/bin/env python3
"""
Codette Combined Application
Integrates web backend and Gradio interface into a unified system
"""

import asyncio
import logging
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional

# ML/DL imports
import torch
import gradio as gr
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
codette_final = project_root / "Codette_final"
sys.path.insert(0, str(codette_final))
sys.path.insert(0, str(codette_final / "components"))

# Core Codette imports
from components.ai_core import AICore
from components.ai_core_system import AICore as AISystem
from codette import Codette
from codette2 import CodetteCQURE
from components.cognitive_processor import CognitiveProcessor
from defense_system import DefenseSystem
from health_monitor import HealthMonitor
from config_manager import EnhancedAIConfig
from fractal import analyze_identity

# Quantum and simulation imports
from quantum_harmonic_framework import quantum_harmonic_dynamics
from codette_quantum_multicore import codette_experiment_task
from codette_meta_3d import *
from codette_timeline_animation import *

# Component imports
from adaptive_learning import AdaptiveLearningEnvironment
from ai_driven_creativity import AIDrivenCreativity
from ethical_governance import EthicalAIGovernance
from sentiment_analysis import EnhancedSentimentAnalyzer
from real_time_data import RealTimeDataIntegrator
from components.search_engine import SearchEngine

# Authentication and security
from cognitive_auth import CognitiveAuthManager
from aegis_integration import AegisBridge
from aegis_integration.config import AEGIS_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodetteCombinedApplication:
    """Combined Codette Application with Web Backend and Gradio Interface"""
    
    def __init__(self):
        logger.info("Initializing Codette Combined Application...")
        self._initialize_core_systems()
        self._initialize_ml_components()
        self._initialize_interface()
        
    def _initialize_core_systems(self):
        """Initialize core Codette systems"""
        try:
            # Load configuration
            self.config = EnhancedAIConfig("config.json")
            logger.info("Configuration loaded successfully")
            
            # Initialize core AI systems
            self.ai_core = AICore()
            logger.info("AI Core initialized")
            
            # Initialize classic Codette
            self.codette_classic = Codette(user_name="WebUser")
            logger.info("Classic Codette initialized")
            
            # Initialize CQURE system
            self.codette_cqure = CodetteCQURE(
                perspectives=["Newton", "DaVinci", "Ethical", "Quantum", "Memory"],
                ethical_considerations="Codette Manifesto: kindness, inclusion, safety, hope.",
                spiderweb_dim=5,
                memory_path="quantum_cocoon.json",
                recursion_depth=4,
                quantum_fluctuation=0.07
            )
            logger.info("Codette CQURE initialized")
            
            # Initialize processing engines
            self.cognitive_processor = CognitiveProcessor(["scientific", "creative", "emotional"])
            self.defense_system = DefenseSystem(["evasion", "adaptability", "barrier"])
            self.health_monitor = HealthMonitor()
            logger.info("Processing engines initialized")
            
            # Initialize components
            self.learning_env = AdaptiveLearningEnvironment()
            self.creativity_engine = AIDrivenCreativity()
            self.ethical_gov = EthicalAIGovernance()
            self.sentiment_analyzer = EnhancedSentimentAnalyzer()
            self.data_integrator = RealTimeDataIntegrator()
            logger.info("Component systems initialized")
            
            # Initialize search engine
            self.search_engine = SearchEngine()
            logger.info("Search engine initialized")
            
        except Exception as e:
            logger.error(f"Core system initialization failed: {e}")
            raise
            
    def _initialize_ml_components(self):
        """Initialize ML components including language model"""
        try:
            model_name = self.config.get("model_name", "gpt2-large")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer initialized successfully")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Using GPU for inference")
            else:
                logger.info("Using CPU for inference")
                
            self.model.eval()
            
            # Set up AI core with model
            self.ai_core.model = self.model
            self.ai_core.tokenizer = self.tokenizer
            self.ai_core.model_id = model_name
            
            # Initialize AEGIS
            self.aegis_bridge = AegisBridge(self.ai_core, AEGIS_CONFIG)
            self.ai_core.set_aegis_bridge(self.aegis_bridge)
            
            logger.info("ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"ML component initialization failed: {e}")
            raise
            
    def _initialize_interface(self):
        """Initialize Gradio interface"""
        self.interface = gr.Blocks(title="Codette", theme=gr.themes.Soft())
        with self.interface:
            gr.Markdown("""# 🤖 Codette
            Your AI programming assistant with chat and search capabilities.""")
            
            with gr.Tabs():
                with gr.Tab("Chat"):
                    self.chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        bubble_full_width=False,
                        avatar_images=("👤", "🤖"),
                        height=500,
                        show_label=False,
                        container=True
                    )
                    
                    with gr.Row():
                        self.txt = gr.Textbox(
                            show_label=False,
                            placeholder="Type your message here...",
                            container=False,
                            scale=8,
                            autofocus=True
                        )
                        self.submit_btn = gr.Button("Send", scale=1, variant="primary")
                    
                    with gr.Row():
                        self.clear_btn = gr.Button("Clear Chat")
                    
                    # Set up chat event handlers
                    self.txt.submit(
                        self.process_message, 
                        [self.txt, self.chatbot], 
                        [self.txt, self.chatbot],
                        api_name="chat_submit",
                        queue=True
                    )
                    
                    self.submit_btn.click(
                        self.process_message, 
                        [self.txt, self.chatbot], 
                        [self.txt, self.chatbot],
                        api_name="chat_button",
                        queue=True
                    )
                    
                    self.clear_btn.click(
                        self.clear_history, 
                        None, 
                        [self.chatbot, self.txt], 
                        queue=False,
                        api_name="clear_chat"
                    )
                    
                with gr.Tab("Search"):
                    gr.Markdown("""### 🔍 Knowledge Search
                    Search through Codette's knowledge base for information about AI, programming, and technology.""")
                    
                    with gr.Row():
                        self.search_input = gr.Textbox(
                            show_label=False,
                            placeholder="Enter your search query...",
                            container=False,
                            scale=8
                        )
                        self.search_btn = gr.Button("Search", scale=1, variant="primary")
                    
                    self.search_output = gr.Markdown()
                    
                    # Set up search event handlers
                    self.search_btn.click(self.sync_search, self.search_input, self.search_output)
                    self.search_input.submit(self.sync_search, self.search_input, self.search_output)
                    
        logger.info("Gradio interface initialized")
    
    async def process_message(self, message: str, history: list) -> tuple:
        """Process chat messages with improved context management"""
        try:
            # Clean input
            message = message.strip()
            if not message:
                return "", history
                
            try:
                # Process through Codette systems
                response = await self.process_query(message)
                
                # Extract main response
                if isinstance(response, dict):
                    final_response = response.get("cqure_response", response.get("classic_response", ""))
                else:
                    final_response = str(response)
                
                # Clean and validate response
                if final_response is None:
                    raise ValueError("Generated response is None")
                    
                if len(final_response) > 1000:
                    final_response = final_response[:997] + "..."
                
                # Update history
                history.append((message, final_response))
                return "", history
                    
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}\n{traceback.format_exc()}")
            error_msg = (
                "I apologize, but I encountered an error processing your request. "
                "Please try again with a different query."
            )
            history.append((message, error_msg))
            return "", history
    
    def clear_history(self):
        """Clear the chat history and AI core memory"""
        self.ai_core.response_memory = []
        return [], []
    
    async def search_knowledge(self, query: str) -> str:
        """Perform a search and return formatted results"""
        try:
            return await self.search_engine.get_knowledge(query)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"I encountered an error while searching: {str(e)}"
    
    def sync_search(self, query: str) -> str:
        """Synchronous wrapper for search function"""
        return asyncio.run(self.search_knowledge(query))
    
    async def process_query(self, query: str, user_id: str = "web_user") -> Dict[str, Any]:
        """Process a query through all Codette systems"""
        try:
            logger.info(f"Processing query: {query}")
            
            # Health check first
            health_status = await self.health_monitor.check_status()
            
            # Sentiment analysis
            sentiment = self.sentiment_analyzer.analyze(query)
            
            # Cognitive processing
            insights = self.cognitive_processor.generate_insights(query)
            
            # Classic Codette response
            classic_response = self.codette_classic.respond(query)
            
            # CQURE response
            cqure_response = self.codette_cqure.answer(query)
            
            # Apply defense filters
            filtered_response = self.defense_system.apply_defenses(cqure_response)
            
            # Ethical governance
            ethical_decision = self.ethical_gov.enforce_policies(filtered_response)
            
            # Compile comprehensive response
            response = {
                "query": query,
                "insights": insights,
                "classic_response": classic_response,
                "cqure_response": filtered_response,
                "ethical_decision": ethical_decision,
                "sentiment": sentiment,
                "health_status": health_status,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "error": f"Processing failed: {str(e)}",
                "query": query,
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def shutdown(self):
        """Cleanup function for graceful shutdown"""
        try:
            # Save final quantum state if available
            if hasattr(self.ai_core, 'cocoon_manager') and self.ai_core.cocoon_manager:
                try:
                    self.ai_core.cocoon_manager.save_cocoon({
                        "type": "shutdown",
                        "quantum_state": self.ai_core.quantum_state
                    })
                    logger.info("Final quantum state saved")
                except Exception as e:
                    logger.error(f"Error saving final quantum state: {e}")
            
            # Shutdown AI core
            try:
                await self.ai_core.shutdown()
                logger.info("AI Core shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down AI Core: {e}")
                
            # Clear CUDA cache if GPU was used
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
                except Exception as e:
                    logger.error(f"Error clearing CUDA cache: {e}")
                    
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise
    
    def run(self):
        """Run the combined application"""
        try:
            # Set up exception handling
            def handle_exception(loop, context):
                msg = context.get("exception", context["message"])
                logger.error(f"Caught exception: {msg}")
                
            # Set up asyncio event loop
            loop = asyncio.new_event_loop()
            loop.set_exception_handler(handle_exception)
            asyncio.set_event_loop(loop)
            
            # Launch Gradio interface
            self.interface.queue().launch(
                prevent_thread_lock=True,
                share=False,
                server_name="127.0.0.1",
                show_error=True
            )
            
            try:
                # Keep the main loop running
                loop.run_forever()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            try:
                loop.run_until_complete(self.shutdown())
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        finally:
            try:
                tasks = asyncio.all_tasks(loop)
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                loop.close()
            except Exception as e:
                logger.error(f"Error closing loop: {e}")
                sys.exit(1)
            sys.exit(0)

# Global application instance
app = None

def get_app() -> CodetteCombinedApplication:
    """Get the global application instance"""
    global app
    if app is None:
        app = CodetteCombinedApplication()
    return app

if __name__ == "__main__":
    try:
        # Initialize and run application
        application = get_app()
        application.run()
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)