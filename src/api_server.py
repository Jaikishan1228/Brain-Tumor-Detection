"""
Flask API server for brain tumor detection
"""

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import json
import os
import io
import base64
from pathlib import Path
import logging

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from models.brain_tumor_model import BrainTumorDetector
from data.data_processor import ImagePreprocessor
from utils.evaluation import GradCAM


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Global variables for model and configuration
model = None
config = None
gradcam = None
preprocessor = ImagePreprocessor()


def load_model(model_path, config_path):
    """Load trained model and configuration"""
    global model, config, gradcam
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        
        # Initialize Grad-CAM
        gradcam = GradCAM(model)
        logger.info("Grad-CAM initialized")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


def preprocess_image(image_data, source_type='file'):
    """Preprocess image data"""
    try:
        if source_type == 'base64':
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif source_type == 'file':
            # Process uploaded file
            image = Image.open(image_data)
        else:
            raise ValueError("Invalid source_type. Use 'base64' or 'file'")
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to model input size
        target_size = (config['image_size'], config['image_size'])
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized, True
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None, False


def predict_image(image):
    """Make prediction on preprocessed image"""
    try:
        # Add batch dimension
        img_batch = np.expand_dims(image, axis=0)
        
        # Get prediction
        prediction = model.predict(img_batch, verbose=0)
        
        # Get class and confidence
        if config['num_classes'] == 2:
            confidence = float(prediction[0][1])
            predicted_class = 1 if confidence > 0.5 else 0
            class_name = config['class_names'][predicted_class]
        else:
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            class_name = config['class_names'][predicted_class]
        
        return {
            'success': True,
            'prediction': {
                'class': int(predicted_class),
                'class_name': class_name,
                'confidence': confidence,
                'probabilities': {
                    config['class_names'][i]: float(prediction[0][i]) 
                    for i in range(len(config['class_names']))
                }
            }
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {'success': False, 'error': str(e)}


def generate_gradcam_visualization(image):
    """Generate Grad-CAM visualization"""
    try:
        heatmap, superimposed = gradcam.generate_heatmap(image)
        
        # Convert to base64 for API response
        _, buffer = cv2.imencode('.png', superimposed)
        gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'success': True,
            'gradcam_image': gradcam_b64
        }
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        return {'success': False, 'error': str(e)}


class HealthCheck(Resource):
    """Health check endpoint"""
    
    def get(self):
        return {
            'status': 'healthy',
            'model_loaded': model is not None,
            'service': 'Brain Tumor Detection API'
        }


class ModelInfo(Resource):
    """Model information endpoint"""
    
    def get(self):
        if model is None or config is None:
            return {'error': 'Model not loaded'}, 500
        
        return {
            'model_info': {
                'architecture': config['model_name'],
                'input_size': f"{config['image_size']}x{config['image_size']}",
                'num_classes': config['num_classes'],
                'class_names': config['class_names'],
                'accuracy': config.get('final_accuracy', 'N/A')
            }
        }


class PredictSingle(Resource):
    """Single image prediction endpoint"""
    
    def post(self):
        if model is None:
            return {'error': 'Model not loaded'}, 500
        
        try:
            # Check if image is provided
            if 'image' not in request.files and 'image_data' not in request.json:
                return {'error': 'No image provided'}, 400
            
            # Process image
            if 'image' in request.files:
                # File upload
                image_file = request.files['image']
                if image_file.filename == '':
                    return {'error': 'No image selected'}, 400
                
                processed_image, success = preprocess_image(image_file, 'file')
            else:
                # Base64 image data
                image_data = request.json['image_data']
                processed_image, success = preprocess_image(image_data, 'base64')
            
            if not success:
                return {'error': 'Failed to process image'}, 400
            
            # Make prediction
            result = predict_image(processed_image)
            
            # Generate Grad-CAM if requested
            include_gradcam = request.args.get('include_gradcam', 'false').lower() == 'true'
            if include_gradcam and result['success']:
                gradcam_result = generate_gradcam_visualization(processed_image)
                if gradcam_result['success']:
                    result['gradcam'] = gradcam_result['gradcam_image']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction endpoint: {str(e)}")
            return {'error': str(e)}, 500


class PredictBatch(Resource):
    """Batch image prediction endpoint"""
    
    def post(self):
        if model is None:
            return {'error': 'Model not loaded'}, 500
        
        try:
            # Get batch of images
            if 'images' not in request.files and 'image_batch' not in request.json:
                return {'error': 'No images provided'}, 400
            
            results = []
            
            if 'images' in request.files:
                # Multiple file uploads
                image_files = request.files.getlist('images')
                
                for i, image_file in enumerate(image_files):
                    if image_file.filename == '':
                        continue
                    
                    processed_image, success = preprocess_image(image_file, 'file')
                    if success:
                        result = predict_image(processed_image)
                        result['image_index'] = i
                        result['filename'] = image_file.filename
                        results.append(result)
                    else:
                        results.append({
                            'success': False,
                            'error': f'Failed to process image {i}',
                            'image_index': i,
                            'filename': image_file.filename
                        })
            else:
                # Batch of base64 images
                image_batch = request.json['image_batch']
                
                for i, image_data in enumerate(image_batch):
                    processed_image, success = preprocess_image(image_data, 'base64')
                    if success:
                        result = predict_image(processed_image)
                        result['image_index'] = i
                        results.append(result)
                    else:
                        results.append({
                            'success': False,
                            'error': f'Failed to process image {i}',
                            'image_index': i
                        })
            
            return {
                'success': True,
                'total_images': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in batch prediction endpoint: {str(e)}")
            return {'error': str(e)}, 500


# Add resources to API
api.add_resource(HealthCheck, '/health')
api.add_resource(ModelInfo, '/model/info')
api.add_resource(PredictSingle, '/predict')
api.add_resource(PredictBatch, '/predict/batch')


@app.route('/')
def index():
    """API documentation endpoint"""
    return {
        'service': 'Brain Tumor Detection API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/model/info': 'GET - Model information',
            '/predict': 'POST - Single image prediction',
            '/predict/batch': 'POST - Batch image prediction'
        },
        'usage': {
            'single_prediction': {
                'method': 'POST',
                'url': '/predict',
                'parameters': {
                    'image': 'multipart/form-data file',
                    'include_gradcam': 'boolean (optional)'
                }
            },
            'batch_prediction': {
                'method': 'POST',
                'url': '/predict/batch',
                'parameters': {
                    'images': 'multiple multipart/form-data files'
                }
            }
        }
    }


@app.errorhandler(404)
def not_found(error):
    return {'error': 'Endpoint not found'}, 404


@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500


def initialize_api(model_path=None, config_path=None):
    """Initialize the API with a model"""
    if model_path and config_path:
        success = load_model(model_path, config_path)
        if success:
            logger.info("API initialized successfully with model")
        else:
            logger.warning("API started without model - predictions will not be available")
    else:
        logger.info("API started without model - use /model/load endpoint to load a model")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain Tumor Detection API Server')
    parser.add_argument('--model_path', type=str, help='Path to trained model file')
    parser.add_argument('--config_path', type=str, help='Path to model configuration file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize API
    initialize_api(args.model_path, args.config_path)
    
    # Run the application
    app.run(host=args.host, port=args.port, debug=args.debug)