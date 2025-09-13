"""
Brain Tumor Detection Model Architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


class BrainTumorDetector:
    """
    Brain Tumor Detection Model using Transfer Learning
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, base_model='resnet50'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model
        self.model = None
        
    def build_model(self, trainable_layers=None):
        """
        Build the model architecture with transfer learning
        
        Args:
            trainable_layers (int): Number of layers to make trainable from the top
        """
        # Define base models
        base_models = {
            'resnet50': ResNet50,
            'vgg16': VGG16,
            'efficientnetb0': EfficientNetB0
        }
        
        if self.base_model_name not in base_models:
            raise ValueError(f"Base model {self.base_model_name} not supported")
        
        # Load pre-trained base model
        base_model = base_models[self.base_model_name](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # If specified, make top layers trainable for fine-tuning
        if trainable_layers:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Build the complete model
        model = keras.Sequential([
            # Preprocessing layers
            layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0),
            
            # Base model
            base_model,
            
            # Custom classification head
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001, metrics=None):
        """
        Compile the model with optimizer and metrics
        
        Args:
            learning_rate (float): Learning rate for the optimizer
            metrics (list): List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall']
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy',
            metrics=metrics
        )
    
    def get_callbacks(self, monitor='val_loss', patience=10):
        """
        Get training callbacks
        
        Args:
            monitor (str): Metric to monitor
            patience (int): Patience for early stopping
        
        Returns:
            list: List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, train_data, validation_data, epochs=50, callbacks=None):
        """
        Train the model
        
        Args:
            train_data: Training data generator
            validation_data: Validation data generator
            epochs (int): Number of epochs
            callbacks (list): Training callbacks
        
        Returns:
            History object
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
    
    def predict(self, image):
        """
        Make prediction on a single image
        
        Args:
            image (np.array): Preprocessed image
        
        Returns:
            tuple: (prediction_class, confidence_score)
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)
        
        if self.num_classes == 2:
            confidence = float(prediction[0][1])
            predicted_class = 1 if confidence > 0.5 else 0
        else:
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
        
        return predicted_class, confidence
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            raise ValueError("No model built")
        return self.model.summary()


def create_model(base_model='resnet50', input_shape=(224, 224, 3), num_classes=2):
    """
    Factory function to create and return a brain tumor detection model
    
    Args:
        base_model (str): Base model architecture
        input_shape (tuple): Input image shape
        num_classes (int): Number of classes
    
    Returns:
        BrainTumorDetector: Configured model instance
    """
    detector = BrainTumorDetector(
        input_shape=input_shape,
        num_classes=num_classes,
        base_model=base_model
    )
    
    detector.build_model()
    detector.compile_model()
    
    return detector