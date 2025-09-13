"""
Streamlit web application for brain tumor detection
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from models.brain_tumor_model import BrainTumorDetector
from data.data_processor import ImagePreprocessor
from utils.evaluation import GradCAM


class BrainTumorWebApp:
    """Streamlit web application for brain tumor detection"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.gradcam = None
        self.preprocessor = ImagePreprocessor()
        
    def load_model(self, model_path, config_path):
        """Load trained model and configuration"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Initialize Grad-CAM
            self.gradcam = GradCAM(self.model)
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess uploaded image"""
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Resize to model input size
        target_size = (self.config['image_size'], self.config['image_size'])
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def predict_image(self, image):
        """Make prediction on preprocessed image"""
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
        
        return {
            'class': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        }
    
    def generate_gradcam(self, image):
        """Generate Grad-CAM visualization"""
        try:
            heatmap, superimposed = self.gradcam.generate_heatmap(image)
            return heatmap, superimposed
        except Exception as e:
            st.error(f"Error generating Grad-CAM: {str(e)}")
            return None, None
    
    def display_prediction_results(self, results, image, original_image):
        """Display prediction results with visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Results")
            
            # Display prediction
            st.write(f"**Predicted Class:** {results['class_name']}")
            st.write(f"**Confidence:** {results['confidence']:.2%}")
            
            # Create confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=results['confidence'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Score (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Class probabilities
            if len(results['probabilities']) > 1:
                st.subheader("Class Probabilities")
                prob_data = {
                    'Class': self.config['class_names'],
                    'Probability': results['probabilities']
                }
                fig = px.bar(prob_data, x='Class', y='Probability',
                           title="Prediction Probabilities by Class")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Image Analysis")
            
            # Display original and processed images
            st.write("**Original Image:**")
            st.image(original_image, use_column_width=True)
            
            # Generate and display Grad-CAM
            st.write("**Grad-CAM Visualization:**")
            with st.spinner("Generating Grad-CAM visualization..."):
                heatmap, superimposed = self.generate_gradcam(image)
                
                if superimposed is not None:
                    st.image(superimposed, use_column_width=True,
                           caption="Areas highlighted in red/yellow contributed most to the prediction")
                else:
                    st.error("Could not generate Grad-CAM visualization")
    
    def display_model_info(self):
        """Display model information"""
        if self.config:
            st.sidebar.subheader("Model Information")
            st.sidebar.write(f"**Architecture:** {self.config['model_name']}")
            st.sidebar.write(f"**Input Size:** {self.config['image_size']}×{self.config['image_size']}")
            st.sidebar.write(f"**Classes:** {', '.join(self.config['class_names'])}")
            st.sidebar.write(f"**Final Accuracy:** {self.config['final_accuracy']:.2%}")
            
            # Model performance metrics
            if 'classification_report' in self.config:
                st.sidebar.subheader("Performance Metrics")
                for class_name in self.config['class_names']:
                    if class_name in self.config['classification_report']:
                        metrics = self.config['classification_report'][class_name]
                        st.sidebar.write(f"**{class_name}:**")
                        st.sidebar.write(f"  Precision: {metrics.get('precision', 0):.3f}")
                        st.sidebar.write(f"  Recall: {metrics.get('recall', 0):.3f}")
                        st.sidebar.write(f"  F1-Score: {metrics.get('f1-score', 0):.3f}")
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Brain Tumor Detection",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 Brain Tumor Detection System")
        st.markdown("""
        Upload a brain MRI image to detect the presence of tumors using deep learning.
        This system uses a trained CNN model to analyze medical images and provide interpretable results.
        """)
        
        # Sidebar for model loading
        st.sidebar.title("Model Configuration")
        
        # Model file upload or selection
        model_source = st.sidebar.radio(
            "Model Source:",
            ["Upload Model Files", "Use Pretrained Model"]
        )
        
        if model_source == "Upload Model Files":
            model_file = st.sidebar.file_uploader(
                "Upload Model File (.h5)",
                type=['h5'],
                help="Upload your trained model file"
            )
            
            config_file = st.sidebar.file_uploader(
                "Upload Config File (.json)",
                type=['json'],
                help="Upload the model configuration file"
            )
            
            if model_file and config_file:
                # Save uploaded files temporarily
                with open("temp_model.h5", "wb") as f:
                    f.write(model_file.getbuffer())
                
                with open("temp_config.json", "wb") as f:
                    f.write(config_file.getbuffer())
                
                # Load model
                if self.load_model("temp_model.h5", "temp_config.json"):
                    st.sidebar.success("✅ Model loaded successfully!")
                    self.display_model_info()
                
                # Clean up temp files
                try:
                    os.remove("temp_model.h5")
                    os.remove("temp_config.json")
                except:
                    pass
        
        else:
            # Look for pretrained models in models directory
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.h5"))
                config_files = list(models_dir.glob("*.json"))
                
                if model_files and config_files:
                    selected_model = st.sidebar.selectbox(
                        "Select Model:",
                        [f.stem for f in model_files]
                    )
                    
                    model_path = models_dir / f"{selected_model}.h5"
                    config_path = models_dir / f"{selected_model}.json"
                    
                    if model_path.exists() and config_path.exists():
                        if st.sidebar.button("Load Selected Model"):
                            if self.load_model(str(model_path), str(config_path)):
                                st.sidebar.success("✅ Model loaded successfully!")
                                self.display_model_info()
                else:
                    st.sidebar.warning("No pretrained models found in 'models' directory")
            else:
                st.sidebar.warning("Models directory not found")
        
        # Main application area
        if self.model is None:
            st.warning("⚠️ Please load a model to start making predictions")
            st.markdown("""
            ### Getting Started:
            1. **Upload Model Files**: Upload your trained model (.h5) and config (.json) files
            2. **Or Use Pretrained**: Select from available pretrained models
            3. **Upload Image**: Upload a brain MRI image for analysis
            4. **View Results**: Get predictions with confidence scores and Grad-CAM visualizations
            
            ### Supported Formats:
            - JPEG, PNG, BMP, TIFF
            - Recommended: High-resolution MRI images
            
            ### Model Information:
            - Uses transfer learning with CNN architectures (ResNet50, VGG16, EfficientNet)
            - Provides binary classification (Tumor/No Tumor)
            - Includes explainable AI with Grad-CAM visualizations
            """)
        else:
            # Image upload and prediction
            st.subheader("📤 Upload Brain MRI Image")
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload a brain MRI image for tumor detection"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Preprocess image
                processed_image = self.preprocess_image(image)
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    results = self.predict_image(processed_image)
                
                # Display results
                self.display_prediction_results(results, processed_image, image)
                
                # Additional information
                st.subheader("ℹ️ Important Notes")
                st.markdown("""
                - **Medical Disclaimer**: This tool is for research and educational purposes only
                - **Not for Diagnosis**: Results should not be used for medical diagnosis
                - **Consult Professionals**: Always consult qualified healthcare professionals
                - **Accuracy**: Model performance may vary with different image qualities and types
                """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Brain Tumor Detection System - Powered by Deep Learning</p>
            <p>⚠️ For research and educational purposes only</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app"""
    app = BrainTumorWebApp()
    app.run()


if __name__ == "__main__":
    main()