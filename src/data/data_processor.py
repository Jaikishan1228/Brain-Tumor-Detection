"""
Data processing utilities for brain tumor detection
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ImagePreprocessor:
    """Image preprocessing utilities for medical images"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.array: Preprocessed image
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def load_dicom_image(self, dicom_path):
        """
        Load and preprocess DICOM medical image
        
        Args:
            dicom_path (str): Path to DICOM file
            
        Returns:
            np.array: Preprocessed image
        """
        try:
            import pydicom
            import SimpleITK as sitk
            
            # Read DICOM file
            dicom = pydicom.dcmread(dicom_path)
            image = dicom.pixel_array
            
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error processing DICOM image {dicom_path}: {str(e)}")
            return None
    
    def enhance_contrast(self, image):
        """Apply contrast enhancement to medical image"""
        # Convert to grayscale for CLAHE
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb.astype(np.float32) / 255.0
    
    def remove_noise(self, image):
        """Remove noise from medical image"""
        # Convert to uint8 for noise removal
        uint8_image = (image * 255).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(uint8_image, (3, 3), 0)
        
        return denoised.astype(np.float32) / 255.0


class DatasetManager:
    """Manage dataset loading and preparation"""
    
    def __init__(self, data_dir, image_size=(224, 224), batch_size=32):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.preprocessor = ImagePreprocessor(target_size=image_size)
    
    def create_data_generators(self, validation_split=0.2, test_split=0.1):
        """
        Create data generators for training, validation, and testing
        
        Args:
            validation_split (float): Fraction of data for validation
            test_split (float): Fraction of data for testing
            
        Returns:
            tuple: (train_gen, val_gen, test_gen)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split + test_split
        )
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split + test_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_test_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def load_dataset_from_folder(self, folder_structure='class_folders'):
        """
        Load dataset from folder structure
        
        Args:
            folder_structure (str): 'class_folders' or 'single_folder'
            
        Returns:
            tuple: (images, labels, class_names)
        """
        images = []
        labels = []
        class_names = []
        
        if folder_structure == 'class_folders':
            # Expected structure: data_dir/class_name/images
            for class_dir in self.data_dir.iterdir():
                if class_dir.is_dir():
                    class_names.append(class_dir.name)
                    class_label = len(class_names) - 1
                    
                    for img_path in class_dir.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                            image = self.preprocessor.load_image(img_path)
                            if image is not None:
                                images.append(image)
                                labels.append(class_label)
        
        return np.array(images), np.array(labels), class_names
    
    def analyze_dataset(self, images, labels, class_names):
        """
        Analyze and visualize dataset statistics
        
        Args:
            images (np.array): Array of images
            labels (np.array): Array of labels
            class_names (list): List of class names
        """
        print("Dataset Analysis")
        print("=" * 50)
        print(f"Total images: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Class names: {class_names}")
        
        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip([class_names[i] for i in unique], counts))
        
        print("\nClass Distribution:")
        for class_name, count in class_distribution.items():
            percentage = (count / len(labels)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Visualize class distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()))
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Show sample images
        self.show_sample_images(images, labels, class_names)
    
    def show_sample_images(self, images, labels, class_names, samples_per_class=3):
        """
        Display sample images from each class
        
        Args:
            images (np.array): Array of images
            labels (np.array): Array of labels
            class_names (list): List of class names
            samples_per_class (int): Number of samples to show per class
        """
        fig, axes = plt.subplots(len(class_names), samples_per_class, 
                                figsize=(15, 5 * len(class_names)))
        
        if len(class_names) == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx, class_name in enumerate(class_names):
            class_images = images[labels == class_idx]
            
            for sample_idx in range(min(samples_per_class, len(class_images))):
                ax = axes[class_idx, sample_idx]
                ax.imshow(class_images[sample_idx])
                ax.set_title(f'{class_name} - Sample {sample_idx + 1}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def prepare_data(data_dir, image_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Convenience function to prepare data for training
    
    Args:
        data_dir (str): Path to data directory
        image_size (tuple): Target image size
        batch_size (int): Batch size for generators
        validation_split (float): Validation split ratio
        
    Returns:
        tuple: (train_generator, val_generator, class_names)
    """
    dataset_manager = DatasetManager(data_dir, image_size, batch_size)
    train_gen, val_gen = dataset_manager.create_data_generators(validation_split)
    
    return train_gen, val_gen, train_gen.class_indices