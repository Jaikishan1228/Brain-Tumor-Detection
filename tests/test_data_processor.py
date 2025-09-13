"""
Unit tests for data processing functionality
"""

import unittest
import numpy as np
import tempfile
import os
import cv2
from pathlib import Path

# Add src directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_processor import ImagePreprocessor, DatasetManager


class TestDataProcessor(unittest.TestCase):
    """Test cases for data processing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # Create temporary directory structure for testing
        self.temp_dir = tempfile.mkdtemp()
        self.setup_test_dataset()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def setup_test_dataset(self):
        """Create a dummy dataset for testing"""
        # Create class directories
        tumor_dir = Path(self.temp_dir) / 'tumor'
        no_tumor_dir = Path(self.temp_dir) / 'no_tumor'
        
        tumor_dir.mkdir()
        no_tumor_dir.mkdir()
        
        # Create dummy images
        for i in range(5):
            # Tumor images
            tumor_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(tumor_dir / f'tumor_{i}.jpg'), tumor_img)
            
            # No tumor images
            no_tumor_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(no_tumor_dir / f'no_tumor_{i}.jpg'), no_tumor_img)
    
    def test_image_loading(self):
        """Test image loading functionality"""
        # Get first image path
        image_path = next(Path(self.temp_dir).rglob('*.jpg'))
        
        # Load image
        processed_img = self.preprocessor.load_image(str(image_path))
        
        # Check image properties
        self.assertIsNotNone(processed_img)
        self.assertEqual(processed_img.shape, (224, 224, 3))
        self.assertEqual(processed_img.dtype, np.float32)
        self.assertGreaterEqual(processed_img.min(), 0.0)
        self.assertLessEqual(processed_img.max(), 1.0)
    
    def test_dataset_manager_initialization(self):
        """Test DatasetManager initialization"""
        dataset_manager = DatasetManager(
            data_dir=self.temp_dir,
            image_size=(224, 224),
            batch_size=2
        )
        
        self.assertEqual(dataset_manager.data_dir, Path(self.temp_dir))
        self.assertEqual(dataset_manager.image_size, (224, 224))
        self.assertEqual(dataset_manager.batch_size, 2)
    
    def test_dataset_loading(self):
        """Test dataset loading from folder structure"""
        dataset_manager = DatasetManager(
            data_dir=self.temp_dir,
            image_size=(224, 224),
            batch_size=2
        )
        
        # Load dataset
        images, labels, class_names = dataset_manager.load_dataset_from_folder()
        
        # Check dataset properties
        self.assertEqual(len(images), 10)  # 5 tumor + 5 no_tumor
        self.assertEqual(len(labels), 10)
        self.assertEqual(len(class_names), 2)
        self.assertIn('tumor', class_names)
        self.assertIn('no_tumor', class_names)
        
        # Check image shapes
        for image in images:
            self.assertEqual(image.shape, (224, 224, 3))
        
        # Check label values
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 2)
        self.assertTrue(all(label in [0, 1] for label in unique_labels))
    
    def test_data_generators(self):
        """Test data generator creation"""
        dataset_manager = DatasetManager(
            data_dir=self.temp_dir,
            image_size=(224, 224),
            batch_size=2
        )
        
        # Create data generators
        train_gen, val_gen = dataset_manager.create_data_generators(
            validation_split=0.3
        )
        
        # Check generators
        self.assertIsNotNone(train_gen)
        self.assertIsNotNone(val_gen)
        
        # Check generator properties
        self.assertEqual(train_gen.target_size, (224, 224))
        self.assertEqual(val_gen.target_size, (224, 224))
        
        # Check that generators produce data
        train_batch = next(train_gen)
        self.assertEqual(len(train_batch), 2)  # images and labels
        
        val_batch = next(val_gen)
        self.assertEqual(len(val_batch), 2)  # images and labels


class TestImagePreprocessorAdvanced(unittest.TestCase):
    """Advanced tests for ImagePreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Create test image
        test_img = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Apply contrast enhancement
        enhanced_img = self.preprocessor.enhance_contrast(test_img)
        self.assertEqual(enhanced_img.shape, test_img.shape)
        
        # Apply noise removal
        denoised_img = self.preprocessor.remove_noise(enhanced_img)
        self.assertEqual(denoised_img.shape, test_img.shape)
        
        # Check value ranges
        self.assertGreaterEqual(denoised_img.min(), 0.0)
        self.assertLessEqual(denoised_img.max(), 1.0)
    
    def test_invalid_image_handling(self):
        """Test handling of invalid image paths"""
        invalid_path = "/path/that/does/not/exist.jpg"
        result = self.preprocessor.load_image(invalid_path)
        self.assertIsNone(result)
    
    def test_different_image_formats(self):
        """Test loading different image formats"""
        formats = ['.jpg', '.png', '.bmp']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for fmt in formats:
                with self.subTest(format=fmt):
                    # Create test image
                    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    image_path = os.path.join(temp_dir, f'test{fmt}')
                    
                    # Save image in different formats
                    if fmt == '.jpg':
                        cv2.imwrite(image_path, test_img)
                    elif fmt == '.png':
                        cv2.imwrite(image_path, test_img)
                    elif fmt == '.bmp':
                        cv2.imwrite(image_path, test_img)
                    
                    # Test loading
                    if os.path.exists(image_path):
                        processed_img = self.preprocessor.load_image(image_path)
                        
                        if processed_img is not None:
                            self.assertEqual(processed_img.shape, (224, 224, 3))
                            self.assertGreaterEqual(processed_img.min(), 0.0)
                            self.assertLessEqual(processed_img.max(), 1.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)