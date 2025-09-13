"""
Training script for brain tumor detection model
"""

import os
import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from models.brain_tumor_model import create_model
from data.data_processor import prepare_data
from utils.evaluation import ModelEvaluator, plot_sample_predictions


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Detection Model')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='resnet50',
                       choices=['resnet50', 'vgg16', 'efficientnetb0'],
                       help='Base model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size (assumes square images)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for saved models')
    parser.add_argument('--tensorboard_logs', type=str, default='logs',
                       help='TensorBoard logs directory')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Enable fine-tuning of pre-trained layers')
    parser.add_argument('--fine_tune_layers', type=int, default=10,
                       help='Number of layers to fine-tune from the top')
    
    return parser.parse_args()


def setup_directories(output_dir, tensorboard_logs):
    """Create necessary directories"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_logs).mkdir(parents=True, exist_ok=True)


def train_model(args):
    """Main training function"""
    print("Starting Brain Tumor Detection Model Training")
    print("=" * 60)
    
    # Setup directories
    setup_directories(args.output_dir, args.tensorboard_logs)
    
    # Prepare data
    print("Preparing data...")
    image_size = (args.image_size, args.image_size)
    
    train_gen, val_gen, class_indices = prepare_data(
        data_dir=args.data_path,
        image_size=image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    # Get class information
    class_names = list(class_indices.keys())
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Create model
    print(f"\nCreating model with base architecture: {args.model_name}")
    model = create_model(
        base_model=args.model_name,
        input_shape=(*image_size, 3),
        num_classes=num_classes
    )
    
    # Compile model with specified learning rate
    model.compile_model(learning_rate=args.learning_rate)
    
    print("\nModel Summary:")
    model.get_model_summary()
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(args.output_dir, f'{args.model_name}_best_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # TensorBoard
    tensorboard = TensorBoard(
        log_dir=args.tensorboard_logs,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    callbacks.append(tensorboard)
    
    # Early stopping and learning rate reduction
    model_callbacks = model.get_callbacks(
        monitor='val_accuracy',
        patience=args.early_stopping_patience
    )
    callbacks.extend(model_callbacks)
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    history = model.train(
        train_data=train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Fine-tuning (if enabled)
    if args.fine_tune:
        print(f"\nStarting fine-tuning with {args.fine_tune_layers} trainable layers...")
        
        # Rebuild model with trainable layers
        fine_tune_model = create_model(
            base_model=args.model_name,
            input_shape=(*image_size, 3),
            num_classes=num_classes
        )
        
        # Load best weights
        fine_tune_model.model.load_weights(checkpoint_path)
        
        # Make top layers trainable
        fine_tune_model.build_model(trainable_layers=args.fine_tune_layers)
        fine_tune_model.compile_model(learning_rate=args.learning_rate / 10)  # Lower learning rate
        
        # Fine-tune with lower learning rate
        fine_tune_checkpoint = os.path.join(args.output_dir, f'{args.model_name}_fine_tuned_model.h5')
        fine_tune_callbacks = [
            ModelCheckpoint(
                fine_tune_checkpoint,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        fine_tune_callbacks.extend(model.get_callbacks(patience=5))
        
        fine_tune_history = fine_tune_model.train(
            train_data=train_gen,
            validation_data=val_gen,
            epochs=args.epochs // 2,  # Fewer epochs for fine-tuning
            callbacks=fine_tune_callbacks
        )
        
        # Use fine-tuned model for evaluation
        model = fine_tune_model
        checkpoint_path = fine_tune_checkpoint
        
        # Combine histories
        for key in history.history:
            if key in fine_tune_history.history:
                history.history[key].extend(fine_tune_history.history[key])
    
    # Load best model for evaluation
    model.load_model(checkpoint_path)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(model.model, class_names)
    
    # Create test generator (using validation data for demo)
    results = evaluator.evaluate_model(val_gen)
    
    # Plot results
    evaluator.plot_confusion_matrix(results['y_true'], results['y_pred_classes'])
    evaluator.plot_roc_curve(results['y_true'], results['y_pred'])
    evaluator.plot_training_history(history)
    
    # Show sample predictions
    plot_sample_predictions(model.model, val_gen, class_names)
    
    # Save training configuration and results
    config = {
        'model_name': args.model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'validation_split': args.validation_split,
        'num_classes': num_classes,
        'class_names': class_names,
        'fine_tune': args.fine_tune,
        'fine_tune_layers': args.fine_tune_layers if args.fine_tune else None,
        'final_accuracy': results['accuracy'],
        'classification_report': results['classification_report']
    }
    
    config_path = os.path.join(args.output_dir, f'{args.model_name}_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Configuration saved to: {config_path}")
    print(f"Final validation accuracy: {results['accuracy']:.4f}")
    
    return model, history, results


def main():
    """Main function"""
    # Set GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Parse arguments and train
    args = parse_arguments()
    
    try:
        model, history, results = train_model(args)
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()