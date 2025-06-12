import unittest
from src.llm.base_model import ImageDescriber
from src.utils.image_utils import ImageProcessor
import os

class TestImageDescriber(unittest.TestCase):
    def setUp(self):
        self.describer = ImageDescriber()
        self.test_image_path = "data/Media.jpg"
    
    def test_image_processing(self):
        # Ensure image exists
        self.assertTrue(os.path.exists(self.test_image_path))
        
        # Test image validation
        self.assertTrue(ImageProcessor.validate_image(self.test_image_path))
        
        # Test image description
        description = self.describer.describe_image(self.test_image_path)
        self.assertIsNotNone(description)
        self.assertIsInstance(description, str)

if __name__ == '__main__':
    unittest.main()