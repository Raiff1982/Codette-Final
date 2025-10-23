import unittest
from unittest.mock import MagicMock, patch
from utils.response_verifier import ResponseVerifier
from typing import Dict, Any

class TestResponseVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = ResponseVerifier()
        # Mock grounding truth verification
        self.verifier.grounding_truth = MagicMock()
        
    def test_verify_insight(self):
        """Test single insight verification"""
        self.verifier.grounding_truth.verify_statement.return_value = {
            "verified": True,
            "confidence": 0.85
        }
        
        insight = "Python is an interpreted language"
        result = self.verifier.verify_insight(insight, "scientific")
        
        self.assertTrue(result["verified"])
        self.assertGreaterEqual(result["confidence"], 0.8)
        self.assertEqual(result["mode"], "scientific")
        
    def test_verify_creative_insight(self):
        """Test creative mode confidence adjustment"""
        self.verifier.grounding_truth.verify_statement.return_value = {
            "verified": False,
            "confidence": 0.6
        }
        
        insight = "Code is like poetry in motion"
        result = self.verifier.verify_insight(insight, "creative")
        
        # Creative mode should boost confidence
        self.assertGreater(result["confidence"], 0.6)
        
    def test_verify_quantum_insight(self):
        """Test quantum mode uncertainty handling"""
        self.verifier.grounding_truth.verify_statement.return_value = {
            "verified": False,
            "confidence": 0.3
        }
        
        insight = "Functions exist in multiple states until observed"
        result = self.verifier.verify_insight(insight, "quantum")
        
        # Quantum mode should have minimum confidence
        self.assertGreaterEqual(result["confidence"], 0.5)
        
    def test_process_multi_perspective_response(self):
        """Test multi-perspective response processing"""
        self.verifier.grounding_truth.verify_statement.side_effect = [
            {"verified": True, "confidence": 0.9},
            {"verified": False, "confidence": 0.6},
            {"verified": True, "confidence": 0.8}
        ]
        
        insights = [
            "Python uses garbage collection",
            "Python code flows like water",
            "Python supports multiple paradigms"
        ]
        modes = ["scientific", "creative", "philosophical"]
        
        result = self.verifier.process_multi_perspective_response(insights, modes)
        
        self.assertEqual(len(result["verified_insights"]), 2)
        self.assertEqual(len(result["uncertain_insights"]), 1)
        self.assertGreater(result["overall_confidence"], 0.7)
        
    def test_qualifiers(self):
        """Test response qualification system"""
        high_conf = self.verifier._get_base_qualifier(0.9)
        med_conf = self.verifier._get_base_qualifier(0.6)
        low_conf = self.verifier._get_base_qualifier(0.3)
        
        self.assertIn("Evidence", high_conf)
        self.assertIn("appears", med_conf)
        self.assertIn("possible", low_conf)
        
        sci_qualifier = self.verifier._get_mode_qualifier("scientific")
        creative_qualifier = self.verifier._get_mode_qualifier("creative")
        
        self.assertIn("data", sci_qualifier)
        self.assertIn("creative", creative_qualifier)
        
    def test_overall_confidence(self):
        """Test overall confidence calculation"""
        verified = [
            {"confidence": 0.9},
            {"confidence": 0.8}
        ]
        uncertain = [
            {"confidence": 0.6},
            {"confidence": 0.4}
        ]
        
        confidence = self.verifier._calculate_overall_confidence(verified, uncertain)
        
        self.assertGreater(confidence, 0.6)
        self.assertLess(confidence, 0.8)
        
if __name__ == '__main__':
    unittest.main()