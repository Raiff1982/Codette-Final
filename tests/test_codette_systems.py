import unittest
import asyncio
import torch
from unittest.mock import MagicMock, patch
from ai_core import AICore
from cognitive_processor import CognitiveProcessor
from utils.cocoon_manager import CocoonManager
from utils.response_verifier import ResponseVerifier

class TestCodetteSystems(unittest.TestCase):
    """Integration tests for Codette's core systems"""
    
    def setUp(self):
        # Set up core components
        self.ai_core = AICore(test_mode=True)
        self.test_model = MagicMock()
        self.test_tokenizer = MagicMock()
        
        # Mock model generation
        self.test_model.generate.return_value = torch.tensor([[1, 2, 3]])
        self.test_tokenizer.decode.return_value = "Test response from model"
        
        # Set up AI Core
        self.ai_core.model = self.test_model
        self.ai_core.tokenizer = self.test_tokenizer
        
        # Set up cognitive processor
        self.ai_core.cognitive_processor = CognitiveProcessor(
            modes=["scientific", "creative", "quantum"],
            quantum_state={"coherence": 0.7}
        )
        
        # Set up cocoon manager
        self.ai_core.cocoon_manager = CocoonManager("./test_cocoons")
        
    def test_component_initialization(self):
        """Test all components are properly initialized"""
        self.assertIsNotNone(self.ai_core.model)
        self.assertIsNotNone(self.ai_core.tokenizer)
        self.assertIsNotNone(self.ai_core.cognitive_processor)
        self.assertIsNotNone(self.ai_core.cocoon_manager)
        
    def test_cognitive_integration(self):
        """Test cognitive processor integration"""
        insights = self.ai_core.cognitive_processor.generate_insights(
            "Test query"
        )
        self.assertIsInstance(insights, dict)
        self.assertIn("insights", insights)
        self.assertGreater(len(insights["insights"]), 0)
        
    def test_quantum_state_flow(self):
        """Test quantum state flows between components"""
        # Update quantum state
        new_state = {"coherence": 0.9}
        self.ai_core.cognitive_processor.quantum_state = new_state
        
        # Save cocoon to trigger state update
        self.ai_core.cocoon_manager.save_cocoon({
            "test": "data",
            "quantum_state": new_state
        })
        
        # Verify state propagation
        current_state = self.ai_core.cocoon_manager.get_latest_quantum_state()
        self.assertEqual(current_state["coherence"], 0.9)
        
    async def test_async_text_generation(self):
        """Test async text generation pipeline"""
        response = await self.ai_core.generate_text_async("Test prompt")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
    def test_response_verification(self):
        """Test response verification system"""
        verifier = self.ai_core.cognitive_processor.verifier
        result = verifier.verify_insight(
            "Python is a programming language",
            "scientific"
        )
        self.assertIsInstance(result, dict)
        self.assertIn("verified", result)
        
    def test_error_handling(self):
        """Test error handling in core components"""
        # Test model error handling
        self.test_model.generate.side_effect = Exception("Test error")
        with self.assertRaises(Exception):
            asyncio.run(self.ai_core.generate_text_async("Test prompt"))
        
        # Reset mock
        self.test_model.generate.side_effect = None
        
    def test_memory_management(self):
        """Test response memory management"""
        # Add responses using proper memory management
        for i in range(6):  # More than memory limit
            self.ai_core._manage_response_memory(f"Response {i}")
            
        # Verify memory limit
        self.assertLessEqual(
            len(self.ai_core.response_memory),
            self.ai_core.response_memory_limit
        )
        
        # Verify we keep the most recent responses
        last_response = self.ai_core.response_memory[-1]
        self.assertEqual(last_response, "Response 5")
        
    @patch('torch.cuda.is_available')
    def test_device_management(self, mock_cuda):
        """Test GPU/CPU management"""
        # Test GPU handling
        mock_cuda.return_value = True
        self.test_model.cuda = MagicMock()
        response = asyncio.run(
            self.ai_core.generate_text_async("Test prompt")
        )
        self.assertIsInstance(response, str)
        
        # Test CPU fallback
        mock_cuda.return_value = False
        response = asyncio.run(
            self.ai_core.generate_text_async("Test prompt")
        )
        self.assertIsInstance(response, str)
        
def run_async_tests():
    """Run async tests with proper event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCodetteSystems)
        
        # Run tests
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    finally:
        loop.close()
        
if __name__ == '__main__':
    success = run_async_tests()
    exit(0 if success else 1)