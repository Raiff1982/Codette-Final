#!/usr/bin/env python3
"""
Codette Web Server Runner
Launches the Flask web interface with all Codette modules imported
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main server entry point"""
    try:
        logger.info("Starting Codette Web Server...")
        
        # Import and run the web interface
        from web_interface import web_app, initialize_codette
        
        # Initialize Codette systems
        logger.info("Initializing Codette AI systems...")
        initialize_codette()
        
        # Start the web server
        logger.info("Starting Flask web server...")
        web_app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()