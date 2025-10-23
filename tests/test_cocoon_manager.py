import unittest
import os
import json
import shutil
from datetime import datetime
from utils.cocoon_manager import CocoonManager

class TestCocoonManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./test_cocoons"
        # Clean up any existing test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        # Create fresh test directory
        os.makedirs(self.test_dir)
        self.manager = CocoonManager(self.test_dir)
        
    def tearDown(self):
        # Clean up test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_save_and_load_cocoon(self):
        """Test saving and loading a cocoon"""
        test_data = {
            "test_key": "test_value",
            "quantum_state": {"coherence": 0.7}
        }
        
        # Save cocoon
        success = self.manager.save_cocoon(test_data)
        self.assertTrue(success)
        
        # Check file was created
        files = os.listdir(self.test_dir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith('.cocoon'))
        
        # Load cocoons
        self.manager.load_cocoons()
        
        # Verify data
        self.assertEqual(len(self.manager.cocoon_data), 1)
        loaded_data = self.manager.cocoon_data[0]['data']
        self.assertEqual(loaded_data['test_key'], "test_value")
        self.assertEqual(loaded_data['quantum_state']['coherence'], 0.7)
        
    def test_quantum_state_management(self):
        """Test quantum state updates and retrieval"""
        initial_state = self.manager.get_latest_quantum_state()
        self.assertEqual(initial_state['coherence'], 0.5)
        
        # Update quantum state
        new_state = {"coherence": 0.8}
        self.manager.update_quantum_state(new_state)
        
        current_state = self.manager.get_latest_quantum_state()
        self.assertEqual(current_state['coherence'], 0.8)
        
    def test_invalid_cocoon_handling(self):
        """Test handling of invalid cocoon files"""
        # Create invalid cocoon file
        invalid_path = os.path.join(self.test_dir, "invalid.cocoon")
        with open(invalid_path, 'w') as f:
            f.write("Invalid JSON")
            
        # Should not raise exception
        self.manager.load_cocoons()
        self.assertEqual(len(self.manager.cocoon_data), 0)
        
    def test_cocoon_validation(self):
        """Test cocoon data validation"""
        invalid_cocoon = {"data": {}}  # Missing timestamp
        self.assertFalse(self.manager._validate_cocoon(invalid_cocoon))
        
        valid_cocoon = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data": {}
        }
        self.assertTrue(self.manager._validate_cocoon(valid_cocoon))
        
    def test_get_latest_cocoons(self):
        """Test retrieving latest cocoons"""
        # Save multiple cocoons
        for i in range(10):
            self.manager.save_cocoon({"test_key": f"value_{i}"})
            
        latest = self.manager.get_latest_cocoons(limit=5)
        self.assertEqual(len(latest), 5)
        self.assertEqual(latest[0]['data']['test_key'], "value_9")
        
if __name__ == '__main__':
    unittest.main()