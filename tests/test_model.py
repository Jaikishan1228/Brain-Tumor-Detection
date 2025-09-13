"""
Unit tests for brain tumor detection model
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.brain_tumor_model import BrainTumorDetector, create_model
from data.data_processor import ImagePreprocessor


class TestBrainTumorModel(unittest.TestCase):
    """Test cases for BrainTumorDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_shape = (224, 224, 3)
        self.num_classes = 2
        self.detector = BrainTumorDetector(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            base_model='resnet50'
        )
    
    def test_model_creation(self):
        """Test model creation"""
        model = self.detector.build_model()
        self.assertIsNotNone(model)
        
        # Check input shape
        self.assertEqual(model.input_shape[1:], self.input_shape)
        
        # Check output shape
        self.assertEqual(model.output_shape[1], self.num_classes)
    
    def test_model_compilation(self):
        """Test model compilation"""
        self.detector.build_model()
        self.detector.compile_model()
        
        # Check if optimizer is set
        self.assertIsNotNone(self.detector.model.optimizer)
    
    def test_prediction_shape(self):
        """Test prediction output shape"""
        self.detector.build_model()
        self.detector.compile_model()
        
        # Create dummy image
        dummy_image = np.random.rand(*self.input_shape)
        
        # Make prediction
        pred_class, confidence = self.detector.predict(dummy_image)
        
        # Check output types and ranges
        self.assertIsInstance(pred_class, int)
        self.assertIsInstance(confidence, float)
        self.assertIn(pred_class, [0, 1])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        self.detector.build_model()
        self.detector.compile_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            
            # Save model
            self.detector.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new detector and load model
            new_detector = BrainTumorDetector(
                input_shape=self.input_shape,
                num_classes=self.num_classes
            )
            new_detector.load_model(model_path)
            
            # Test that loaded model works
            dummy_image = np.random.rand(*self.input_shape)
            pred_class, confidence = new_detector.predict(dummy_image)
            
            self.assertIsInstance(pred_class, int)
            self.assertIsInstance(confidence, float)


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    def test_create_dummy_image(self):
        """Test creation of dummy image for testing"""
        # Create a dummy image file
        import cv2
        
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, 'test_image.jpg')
            
            # Create dummy image
            dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(image_path, dummy_img)
            
            # Test preprocessing
            processed_img = self.preprocessor.load_image(image_path)
            
            self.assertIsNotNone(processed_img)
            self.assertEqual(processed_img.shape, (224, 224, 3))
            self.assertGreaterEqual(processed_img.min(), 0.0)
            self.assertLessEqual(processed_img.max(), 1.0)
    
    def test_contrast_enhancement(self):
        """Test contrast enhancement"""
        # Create dummy image
        dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Apply contrast enhancement
        enhanced_img = self.preprocessor.enhance_contrast(dummy_img)
        
        self.assertEqual(enhanced_img.shape, dummy_img.shape)
        self.assertGreaterEqual(enhanced_img.min(), 0.0)
        self.assertLessEqual(enhanced_img.max(), 1.0)
    
    def test_noise_removal(self):
        """Test noise removal"""
        # Create dummy image
        dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Apply noise removal
        denoised_img = self.preprocessor.remove_noise(dummy_img)
        
        self.assertEqual(denoised_img.shape, dummy_img.shape)
        self.assertGreaterEqual(denoised_img.min(), 0.0)
        self.assertLessEqual(denoised_img.max(), 1.0)


class TestModelFactory(unittest.TestCase):
    """Test cases for model factory function"""
    
    def test_create_model_function(self):
        """Test the create_model factory function"""
        model = create_model(
            base_model='resnet50',
            input_shape=(224, 224, 3),
            num_classes=2
        )
        
        self.assertIsInstance(model, BrainTumorDetector)
        self.assertIsNotNone(model.model)
    
    def test_different_architectures(self):
        """Test different model architectures"""
        architectures = ['resnet50', 'vgg16', 'efficientnetb0']
        
        for arch in architectures:
            with self.subTest(architecture=arch):
                model = create_model(
                    base_model=arch,
                    input_shape=(224, 224, 3),
                    num_classes=2
                )
                
                self.assertIsInstance(model, BrainTumorDetector)
                self.assertEqual(model.base_model_name, arch)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)