import unittest
import asyncio
import gradio as gr
from unittest.mock import MagicMock, patch
from codette_interface import create_interface

class TestUIIntegration(unittest.TestCase):
    """Test Codette's unified interface"""
    
    def setUp(self):
        """Set up test environment"""
        self.interface = create_interface()
        self.mock_response = {
            "response": "Test response from Codette",
            "metrics": {"confidence": 0.95},
            "insights": ["Test insight"]
        }
        
        # Mock Codette's respond method
        self.interface.codette.respond = MagicMock(
            return_value=self.mock_response
        )
        
    def test_chat_processing(self):
        """Test chat message processing"""
        message = "Test message"
        history = []
        
        # Process message
        new_text, new_history = self.interface.process_message(message, history)
        
        # Verify response
        self.assertEqual(new_text, "")
        self.assertEqual(len(new_history), 1)
        self.assertEqual(new_history[0][0], message)
        self.assertIn(self.mock_response["response"], new_history[0][1])
        self.assertIn("Metrics", new_history[0][1])
        self.assertIn("Insights", new_history[0][1])
        
    def test_api_processing(self):
        """Test API message processing"""
        message = "Test message"
        
        # Process through API interface
        result = self.interface.process_message(message)
        
        # Verify response format
        self.assertIn("response", result)
        self.assertIn("timestamp", result)
        self.assertIn("metrics", result)
        self.assertIn("insights", result)
        
    def test_search_functionality(self):
        """Test knowledge search"""
        query = "test query"
        result = self.interface.search_knowledge(query)
        self.assertEqual(result, self.mock_response["response"])
        
    def test_system_state(self):
        """Test system state retrieval"""
        state = self.interface.get_system_state()
        self.assertIsInstance(state, str)
        self.assertIn("System State", state)
        self.assertIn("Active Perspectives", state)
        
    @patch('gradio.Blocks')
    def test_gradio_interface(self, mock_blocks):
        """Test Gradio interface creation"""
        # Create mock interface
        mock_interface = MagicMock()
        mock_blocks.return_value = mock_interface
        
        # Create Gradio interface
        interface = self.interface.create_gradio_interface()
        
        # Verify interface creation
        self.assertIsNotNone(interface)
        
    def test_web_interface(self):
        """Test web interface creation"""
        app = self.interface.create_web_app()
        self.assertIsNotNone(app)
        
    def test_error_handling(self):
        """Test error handling"""
        # Simulate error in Codette
        self.interface.codette.respond = MagicMock(
            side_effect=Exception("Test error")
        )
        
        message = "Test message"
        history = []
        
        # Test Gradio interface error handling
        new_text, new_history = self.interface.process_message(message, history)
        self.assertEqual(new_text, "")
        self.assertEqual(len(new_history), 1)
        self.assertIn("error", new_history[0][1].lower())
        
        # Test API interface error handling
        result = self.interface.process_message(message)
        self.assertIn("error", result)
        
def run_tests():
    """Run all tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUIIntegration)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)