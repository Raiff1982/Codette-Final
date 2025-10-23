"""
Test script to validate core Codette functionality
"""
import sys
import os
import importlib.util
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_import(module_name):
    """Try to import a module and check if it's working."""
    try:
        if importlib.util.find_spec(module_name):
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported {module_name}")
            return module
        else:
            logger.error(f"Could not find spec for {module_name}")
            return None
    except Exception as e:
        logger.error(f"Error importing {module_name}: {str(e)}")
        return None

def check_package_data(module, data_path):
    """Try to access package data."""
    try:
        if hasattr(module, 'data') and hasattr(module.data, 'find'):
            path = module.data.find(data_path)
            logger.info(f"Found package data at: {path}")
            return True
    except Exception as e:
        logger.error(f"Error accessing package data {data_path}: {str(e)}")
        return False

def test_fallback_models():
    """Test the fallback model system."""
    try:
        from models.fallback import fallback_manager
        
        # Test NLP model
        nlp_model = fallback_manager.get_model('nlp_transformer')
        logger.info("NLP model loaded successfully" if nlp_model else "NLP model failed")
        
        # Test vision model
        vision_model = fallback_manager.get_model('vision_analyzer')
        logger.info("Vision model loaded successfully" if vision_model else "Vision model failed")
        
        # Test quantum approximation
        quantum_model = fallback_manager.get_model('quantum_simulator')
        logger.info("Quantum approximation loaded successfully" if quantum_model else "Quantum approximation failed")
        
    except Exception as e:
        logger.error(f"Error testing fallback models: {str(e)}")

def main():
    logger.info("=== Codette Test Script ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # List current directory contents
    logger.info("Current directory contents:")
    for item in os.listdir('.'):
        logger.info(f"  {item}")

    # Test imports
    modules_to_test = [
        'codette',
        'arviz',
        'nltk',
        'numpy',
        'scipy',
        'pandas',
        'torch',
        'transformers'
    ]
    
    # Test fallback models
    test_fallback_models()

    for module_name in modules_to_test:
        module = check_import(module_name)
        
        if module_name == 'nltk' and module:
            check_package_data(module, 'corpora/wordnet')

if __name__ == '__main__':
    main()
