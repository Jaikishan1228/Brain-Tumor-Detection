"""
Batch prediction script for brain tumor detection
"""

import os
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tqdm import tqdm

# Import custom modules
from models.brain_tumor_model import BrainTumorDetector
from data.data_processor import ImagePreprocessor
from utils.evaluation import GradCAM


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch Prediction for Brain Tumor Detection')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.h5)')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration file (.json)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save prediction results')
    parser.add_argument('--output_format', type=str, default='csv',
                       choices=['csv', 'json'],
                       help='Output format for results')
    parser.add_argument('--include_gradcam', action='store_true',
                       help='Generate Grad-CAM visualizations')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for predictions')
    parser.add_argument('--image_extensions', nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                       help='Image file extensions to process')
    parser.add_argument('--recursive', action='store_true',
                       help='Process images recursively in subdirectories')
    
    return parser.parse_args()


class BatchPredictor:
    """Batch prediction handler for brain tumor detection"""
    
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.gradcam = None
        self.preprocessor = ImagePreprocessor()
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and configuration"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load configuration
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Configuration loaded from {self.config_path}")
            
            # Initialize Grad-CAM
            self.gradcam = GradCAM(self.model)
            print("✅ Grad-CAM initialized")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def find_images(self, input_dir, extensions, recursive=False):
        """Find all image files in the input directory"""
        input_path = Path(input_dir)
        image_files = []
        
        if recursive:
            for ext in extensions:
                image_files.extend(input_path.rglob(f"*{ext}"))
                image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Resize to model input size
            target_size = (self.config['image_size'], self.config['image_size'])
            img_resized = cv2.resize(img_array, target_size)
            
            # Normalize
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            return img_normalized, True
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            return None, False
    
    def predict_single(self, image):
        """Make prediction on a single preprocessed image"""
        try:
            # Add batch dimension
            img_batch = np.expand_dims(image, axis=0)
            
            # Get prediction
            prediction = self.model.predict(img_batch, verbose=0)
            
            # Get class and confidence
            if self.config['num_classes'] == 2:
                confidence = float(prediction[0][1])
                predicted_class = 1 if confidence > 0.5 else 0
                class_name = self.config['class_names'][predicted_class]
            else:
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                class_name = self.config['class_names'][predicted_class]
            
            # Get all class probabilities
            probabilities = {
                self.config['class_names'][i]: float(prediction[0][i])
                for i in range(len(self.config['class_names']))
            }
            
            return {
                'success': True,
                'class': int(predicted_class),
                'class_name': class_name,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_batch(self, images):
        """Make predictions on a batch of images"""
        try:
            # Stack images into batch
            batch = np.stack(images)
            
            # Get predictions
            predictions = self.model.predict(batch, verbose=0)
            
            results = []
            for i, prediction in enumerate(predictions):
                if self.config['num_classes'] == 2:
                    confidence = float(prediction[1])
                    predicted_class = 1 if confidence > 0.5 else 0
                    class_name = self.config['class_names'][predicted_class]
                else:
                    predicted_class = np.argmax(prediction)
                    confidence = float(prediction[predicted_class])
                    class_name = self.config['class_names'][predicted_class]
                
                probabilities = {
                    self.config['class_names'][j]: float(prediction[j])
                    for j in range(len(self.config['class_names']))
                }
                
                results.append({
                    'success': True,
                    'class': int(predicted_class),
                    'class_name': class_name,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
            
            return results
            
        except Exception as e:
            return [{'success': False, 'error': str(e)} for _ in range(len(images))]
    
    def generate_gradcam(self, image, output_path):
        """Generate and save Grad-CAM visualization"""
        try:
            heatmap, superimposed = self.gradcam.generate_heatmap(image)
            
            # Save Grad-CAM visualization
            cv2.imwrite(str(output_path), superimposed)
            return True
            
        except Exception as e:
            print(f"❌ Error generating Grad-CAM: {str(e)}")
            return False
    
    def process_images(self, input_dir, output_dir, batch_size=32, include_gradcam=False,
                      extensions=None, recursive=False):
        """Process all images in the input directory"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if include_gradcam:
            gradcam_dir = output_path / 'gradcam'
            gradcam_dir.mkdir(exist_ok=True)
        
        # Find all image files
        image_files = self.find_images(input_dir, extensions, recursive)
        print(f"Found {len(image_files)} images to process")
        
        if not image_files:
            print("❌ No images found to process")
            return []
        
        results = []
        processed_images = []
        processed_paths = []
        
        # Process images in batches
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                batch_images = []
                batch_paths = []
                
                # Preprocess batch
                for image_path in batch_files:
                    processed_image, success = self.preprocess_image(image_path)
                    if success:
                        batch_images.append(processed_image)
                        batch_paths.append(image_path)
                    else:
                        # Add failed result
                        results.append({
                            'image_path': str(image_path),
                            'filename': image_path.name,
                            'success': False,
                            'error': 'Preprocessing failed'
                        })
                
                # Make predictions on batch
                if batch_images:
                    if len(batch_images) == 1:
                        # Single prediction
                        batch_results = [self.predict_single(batch_images[0])]
                    else:
                        # Batch prediction
                        batch_results = self.predict_batch(batch_images)
                    
                    # Process results
                    for j, (result, image_path, image) in enumerate(zip(batch_results, batch_paths, batch_images)):
                        result['image_path'] = str(image_path)
                        result['filename'] = image_path.name
                        
                        # Generate Grad-CAM if requested
                        if include_gradcam and result['success']:
                            gradcam_filename = f"{image_path.stem}_gradcam.png"
                            gradcam_path = gradcam_dir / gradcam_filename
                            
                            if self.generate_gradcam(image, gradcam_path):
                                result['gradcam_path'] = str(gradcam_path)
                        
                        results.append(result)
                
                pbar.update(len(batch_files))
        
        print(f"✅ Processed {len(results)} images")
        return results
    
    def save_results(self, results, output_path, format='csv'):
        """Save prediction results to file"""
        output_file = Path(output_path)
        
        if format == 'csv':
            output_file = output_file.with_suffix('.csv')
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                if results:
                    # Flatten probabilities for CSV
                    fieldnames = ['image_path', 'filename', 'success', 'class', 'class_name', 'confidence']
                    
                    # Add probability columns
                    if results[0].get('probabilities'):
                        for class_name in self.config['class_names']:
                            fieldnames.append(f'prob_{class_name}')
                    
                    if results[0].get('gradcam_path'):
                        fieldnames.append('gradcam_path')
                    
                    if results[0].get('error'):
                        fieldnames.append('error')
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for result in results:
                        row = {k: v for k, v in result.items() if k != 'probabilities'}
                        
                        # Add probability columns
                        if result.get('probabilities'):
                            for class_name, prob in result['probabilities'].items():
                                row[f'prob_{class_name}'] = prob
                        
                        writer.writerow(row)
        
        elif format == 'json':
            output_file = output_file.with_suffix('.json')
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump({
                    'model_info': {
                        'model_path': self.model_path,
                        'config_path': self.config_path,
                        'architecture': self.config['model_name'],
                        'class_names': self.config['class_names']
                    },
                    'results': results
                }, jsonfile, indent=2)
        
        print(f"✅ Results saved to {output_file}")
        return str(output_file)


def main():
    """Main function"""
    args = parse_arguments()
    
    print("Brain Tumor Detection - Batch Prediction")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = BatchPredictor(args.model_path, args.config_path)
        
        # Process images
        results = predictor.process_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            include_gradcam=args.include_gradcam,
            extensions=args.image_extensions,
            recursive=args.recursive
        )
        
        # Save results
        output_filename = f"predictions_{Path(args.input_dir).name}"
        output_path = Path(args.output_dir) / output_filename
        
        predictor.save_results(results, output_path, args.output_format)
        
        # Print summary
        successful_predictions = sum(1 for r in results if r.get('success', False))
        failed_predictions = len(results) - successful_predictions
        
        print(f"\n📊 Prediction Summary:")
        print(f"   Total images: {len(results)}")
        print(f"   Successful: {successful_predictions}")
        print(f"   Failed: {failed_predictions}")
        
        if successful_predictions > 0:
            # Class distribution
            class_counts = {}
            for result in results:
                if result.get('success') and result.get('class_name'):
                    class_name = result['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"\n🎯 Prediction Distribution:")
            for class_name, count in class_counts.items():
                percentage = (count / successful_predictions) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n✅ Batch prediction completed successfully!")
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()