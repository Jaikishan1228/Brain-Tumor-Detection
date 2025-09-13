"""
Utility functions for model evaluation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2


class ModelEvaluator:
    """Comprehensive model evaluation utilities"""
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate_model(self, test_data, verbose=True):
        """
        Comprehensive model evaluation
        
        Args:
            test_data: Test data generator
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(test_data, verbose=0)
        y_true = test_data.classes
        
        # Convert predictions to class labels
        if self.num_classes == 2:
            y_pred_classes = (y_pred[:, 1] > 0.5).astype(int)
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred_classes == y_true)
        
        # Classification report
        report = classification_report(
            y_true, y_pred_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        if verbose:
            print("Model Evaluation Results")
            print("=" * 50)
            print(f"Overall Accuracy: {accuracy:.4f}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred_classes, target_names=self.class_names))
        
        # Store results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_classes': y_pred_classes
        }
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred_classes, figsize=(10, 8)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred, figsize=(10, 8)):
        """Plot ROC curve for binary or multiclass classification"""
        plt.figure(figsize=figsize)
        
        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multiclass classification
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                        label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history, figsize=(15, 5)):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[2].plot(history.history['lr'], label='Learning Rate')
            axes[2].set_title('Learning Rate')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_yscale('log')
            axes[2].legend()
            axes[2].grid(True)
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
    
    def _find_target_layer(self):
        """Find the last convolutional layer"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    
    def generate_heatmap(self, image, class_idx=None, alpha=0.4):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image (np.array): Input image
            class_idx (int): Target class index (if None, uses predicted class)
            alpha (float): Transparency for overlay
            
        Returns:
            tuple: (heatmap, superimposed_img)
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            img_array = np.expand_dims(image, axis=0)
        else:
            img_array = image
        
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, class_idx]
        
        # Extract the gradients of the top predicted class with regard to the output feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pool the gradients over all the axes leaving only the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array by "how important this channel is"
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match input image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original image to uint8
        if image.max() <= 1.0:
            image_uint8 = np.uint8(255 * image)
        else:
            image_uint8 = np.uint8(image)
        
        # Superimpose the heatmap on original image
        superimposed_img = cv2.addWeighted(image_uint8, alpha, heatmap, 1 - alpha, 0)
        
        return heatmap, superimposed_img
    
    def visualize_activation(self, image, class_idx=None, figsize=(15, 5)):
        """
        Visualize Grad-CAM activation
        
        Args:
            image (np.array): Input image
            class_idx (int): Target class index
            figsize (tuple): Figure size
        """
        heatmap, superimposed = self.generate_heatmap(image, class_idx)
        
        # Get prediction
        if len(image.shape) == 3:
            img_array = np.expand_dims(image, axis=0)
        else:
            img_array = image
        
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Display results
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        if image.max() <= 1.0:
            axes[0].imshow(image)
        else:
            axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Superimposed
        axes[2].imshow(superimposed)
        axes[2].set_title(f'Grad-CAM Overlay\nPredicted: Class {predicted_class}\nConfidence: {confidence:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return heatmap, superimposed


def plot_sample_predictions(model, test_data, class_names, num_samples=9):
    """
    Plot sample predictions with confidence scores
    
    Args:
        model: Trained model
        test_data: Test data generator
        class_names (list): List of class names
        num_samples (int): Number of samples to show
    """
    # Get batch of test images
    batch_images, batch_labels = next(test_data)
    
    # Get predictions
    predictions = model.predict(batch_images[:num_samples], verbose=0)
    
    # Create subplot
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        # Get true and predicted labels
        true_label = np.argmax(batch_labels[i])
        pred_label = np.argmax(predictions[i])
        confidence = predictions[i][pred_label]
        
        # Display image
        axes[i].imshow(batch_images[i])
        
        # Set title with prediction info
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nConf: {confidence:.3f}'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    # Hide remaining subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def save_evaluation_report(results, model_name, save_path):
    """
    Save comprehensive evaluation report
    
    Args:
        results (dict): Evaluation results
        model_name (str): Name of the model
        save_path (str): Path to save the report
    """
    report = f"""
Brain Tumor Detection Model Evaluation Report
============================================

Model: {model_name}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Performance:
- Accuracy: {results['accuracy']:.4f}

Detailed Metrics by Class:
"""
    
    for class_name, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            report += f"\n{class_name}:"
            report += f"\n  Precision: {metrics.get('precision', 'N/A'):.4f}"
            report += f"\n  Recall: {metrics.get('recall', 'N/A'):.4f}"
            report += f"\n  F1-Score: {metrics.get('f1-score', 'N/A'):.4f}"
            report += f"\n  Support: {metrics.get('support', 'N/A')}"
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Evaluation report saved to: {save_path}")