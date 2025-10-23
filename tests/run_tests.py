import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from test_integration import test_model_integration

class ModelIntegrationTest(unittest.TestCase):
    def test_full_integration(self):
        """Run the full integration test."""
        self.assertTrue(test_model_integration(), "Model integration test failed")

if __name__ == '__main__':
    unittest.main()