# Test script for fallback models
import sys
import logging
from pathlib import Path
import torch
import transformers
from models.fallback import fallback_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_nlp_fallback():
    """Test NLP fallback model."""
    model = fallback_manager.get_model('nlp_transformer')
    if model is None:
        return False
        
    # Test basic functionality
    try:
        tokenizer = model['tokenizer']
        model_obj = model['model']
        
        # Test tokenization
        text = "Testing the NLP model."
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model_obj(**inputs)
        
        return outputs.last_hidden_state is not None
    except Exception as e:
        logger.error(f"NLP test failed: {str(e)}")
        return False

def test_vision_fallback():
    """Test vision fallback model."""
    model = fallback_manager.get_model('vision_analyzer')
    if model is None:
        return False
        
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        return output is not None
    except Exception as e:
        logger.error(f"Vision test failed: {str(e)}")
        return False

def test_quantum_fallback():
    """Test quantum approximation."""
    model = fallback_manager.get_model('quantum_simulator')
    if model is None:
        return False
        
    try:
        # Test with dummy quantum state
        dummy_input = torch.randn(1, 64)
        output = model(dummy_input)
        
        return output is not None
    except Exception as e:
        logger.error(f"Quantum test failed: {str(e)}")
        return False

def main():
    logger.info("=== Testing Fallback Models ===")
    
    # Run tests
    nlp_success = test_nlp_fallback()
    vision_success = test_vision_fallback()
    quantum_success = test_quantum_fallback()
    
    # Report results
    logger.info("\nTest Results:")
    logger.info(f"NLP Fallback: {'✓' if nlp_success else '✗'}")
    logger.info(f"Vision Fallback: {'✓' if vision_success else '✗'}")
    logger.info(f"Quantum Approximation: {'✓' if quantum_success else '✗'}")
    
    # Overall status
    success = all([nlp_success, vision_success, quantum_success])
    logger.info(f"\nOverall Status: {'PASS' if success else 'FAIL'}")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
