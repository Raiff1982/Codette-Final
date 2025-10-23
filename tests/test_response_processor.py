import unittest
from unittest.mock import MagicMock, patch
from utils.response_processor import ResponseProcessor

class TestResponseProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ResponseProcessor()
        
    def test_split_into_statements(self):
        """Test statement splitting logic"""
        test_response = (
            "Python is a programming language. It supports both OOP and functional "
            "programming, and it's great for data science. But performance can vary."
        )
        
        statements = self.processor._split_into_statements(test_response)
        
        self.assertEqual(len(statements), 4)
        self.assertIn("Python is a programming language", statements)
        
    def test_add_qualifier(self):
        """Test confidence qualifier addition"""
        statement = "Python is fast"
        
        high_conf = self.processor._add_qualifier(statement, 0.9)
        med_conf = self.processor._add_qualifier(statement, 0.7)
        low_conf = self.processor._add_qualifier(statement, 0.3)
        
        self.assertIn("highly likely", high_conf)
        self.assertIn("probably", med_conf)
        self.assertIn("uncertain", low_conf)
        
    def test_process_response(self):
        """Test full response processing"""
        # Mock grounding truth verification
        self.processor.grounding_truth.verify_statement = MagicMock()
        self.processor.grounding_truth.verify_statement.side_effect = [
            {"verified": True, "confidence": 0.9},
            {"verified": False, "confidence": 0.6}
        ]
        
        query = "Tell me about Python"
        response = "Python is interpreted. Python is the fastest language."
        
        processed = self.processor.process_response(query, response)
        
        self.assertIn("Here's what I know for certain", processed)
        self.assertIn("Additionally", processed)
        self.assertIn("probably", processed)
        
    def test_context_update(self):
        """Test context history management"""
        query = "Test query"
        response = "Test response"
        
        self.processor.update_context(query, response)
        self.assertEqual(len(self.processor.context_history), 1)
        
        # Test history limit
        for i in range(15):
            self.processor.update_context(f"Query {i}", f"Response {i}")
            
        self.assertLessEqual(len(self.processor.context_history), 10)
        
if __name__ == '__main__':
    unittest.main()