import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import os
import logging
from typing import Optional
import asyncio
import better_ui
from ai_core_system_new import AICore

class CodetteLauncher:
    def __init__(self):
        self.setup_logging()
        self.core: Optional[AICore] = None
        self.ui: Optional[better_ui.CodetteUI] = None
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('codette.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize_core(self):
        """Initialize the AI core system"""
        try:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "config",
                "ai_assistant_config.json"
            )
            
            # Create default config if it doesn't exist
            if not os.path.exists(config_path):
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    f.write('{"elements": []}')
            
            self.core = AICore(config_path)
            self.logger.info("AI Core initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Core: {e}")
            return False
    
    def _process_message(self, message: str):
        """Process a message through the AI core"""
        async def _process():
            try:
                if self.core:
                    result = await self.core.process_input(message)
                    if result.get("status") == "success":
                        return result.get("response", "I processed that, but I'm not sure how to respond.")
                    else:
                        return "I encountered an error processing your request."
                else:
                    return "AI Core is not initialized properly."
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                return "Sorry, I encountered an error processing your request."
        
        # Run async function in event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_process())
    
    def start(self):
        """Start the Codette system"""
        try:
            # Initialize event loop
            loop = asyncio.get_event_loop()
            
            # Initialize core
            if not loop.run_until_complete(self.initialize_core()):
                self.logger.error("Failed to initialize AI Core")
                return
            
            # Create and configure UI
            self.ui = better_ui.CodetteUI()
            
            # Override the UI's _receive_response method to use our AI core
            def new_receive_response(user_message: str):
                response = self._process_message(user_message)
                self.ui.chat_display.insert('end', f"Codette: {response}\n\n", "assistant")
                self.ui.chat_display.see('end')
                self.ui._update_status("Ready")
            
            self.ui._receive_response = new_receive_response
            
            # Start the UI
            self.ui.mainloop()
            
        except Exception as e:
            self.logger.error(f"Error starting Codette: {e}")
            raise

def main():
    """Main entry point"""
    try:
        launcher = CodetteLauncher()
        launcher.start()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
