"""
Artificial Neural Network model for spam detection
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
    GaussianNoise, Activation
)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.utils import plot_model
import numpy as np
import yaml
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ANNModel:
    """
    Advanced ANN model for email spam classification
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize ANN model with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model parameters
        model_config = self.config['model']
        self.input_dim = model_config['input_dim']
        self.hidden_layers = model_config['hidden_layers']
        self.activation = model_config['activation']
        self.output_activation = model_config['output_activation']
        self.dropout_rate = model_config['dropout_rate']
        self.l2_regularization = model_config['l2_regularization']
        self.batch_normalization = model_config['batch_normalization']
        
        # Training parameters
        training_config = self.config['training']
        self.batch_size = training_config['batch_size']
        self.epochs = training_config['epochs']
        self.learning_rate = training_config['learning_rate']
        self.optimizer_name = training_config['optimizer']
        self.loss = training_config['loss']
        self.metrics = training_config['metrics']
        self.validation_split = training_config['validation_split']
        
        # Paths
        paths_config = self.config['paths']
        self.models_dir = Path(paths_config['models'])
        self.logs_dir = Path(paths_config['logs'])
        self.plots_dir = Path(paths_config['plots'])
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build ANN model architecture
        
        Returns:
            Compiled Keras model
        """
        self.logger.info("Building ANN model...")
        self.logger.info(f"Input dimension: {self.input_dim}")
        self.logger.info(f"Hidden layers: {self.hidden_layers}")
        self.logger.info(f"Activation: {self.activation}")
        
        # Create model
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.input_dim,)))
        
        # Add Gaussian noise for regularization (optional)
        model.add(GaussianNoise(0.01))
        
        # Add hidden layers
        for i, units in enumerate(self.hidden_layers):
            # Dense layer
            model.add(Dense(
                units,
                kernel_regularizer=l2(self.l2_regularization),
                bias_regularizer=l2(self.l2_regularization),
                name=f'dense_{i}'
            ))
            
            # Batch normalization
            if self.batch_normalization:
                model.add(BatchNormalization(name=f'batch_norm_{i}'))
            
            # Activation
            model.add(Activation(self.activation, name=f'activation_{i}'))
            
            # Dropout
            model.add(Dropout(self.dropout_rate, name=f'dropout_{i}'))
        
        # Output layer
        model.add(Dense(
            1,
            activation=self.output_activation,
            kernel_regularizer=l2(self.l2_regularization),
            name='output'
        ))
        
        # Compile model
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        
        self.model = model
        self.logger.info("Model built successfully")
        self.logger.info(f"Model summary:\n{model.summary()}")
        
        return model
    
    def _get_optimizer(self):
        """
        Get optimizer based on configuration
        
        Returns:
            Keras optimizer
        """
        optimizer_name = self.optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return Adam(learning_rate=self.learning_rate)
        elif optimizer_name == 'sgd':
            return SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=self.learning_rate)
        else:
            self.logger.warning(f"Unknown optimizer: {optimizer_name}. Using Adam.")
            return Adam(learning_rate=self.learning_rate)
    
    def get_callbacks(self):
        """
        Get training callbacks
        
        Returns:
            List of callbacks
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self.config['model']['name']
        
        # Model checkpoint
        checkpoint_path = self.models_dir / f'{model_name}_{timestamp}.h5'
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config['training']['reduce_lr_factor'],
            patience=self.config['training']['reduce_lr_patience'],
            min_lr=1e-6,
            verbose=1
        )
        
        # TensorBoard
        tensorboard_dir = self.logs_dir / f'tensorboard_{timestamp}'
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        
        # CSV Logger
        csv_logger = CSVLogger(
            self.logs_dir / f'training_{timestamp}.csv',
            separator=',',
            append=False
        )
        
        return [checkpoint, early_stopping, reduce_lr, tensorboard, csv_logger]
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        self.logger.info("Starting model training...")
        
        if self.model is None:
            self.build_model()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            self.logger.info(f"Using validation data: {X_val.shape[0]} samples")
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        self.logger.info(f"Training parameters:")
        self.logger.info(f"  Batch size: {self.batch_size}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Learning rate: {self.learning_rate}")
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split if validation_data is None else 0.0,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Training completed successfully")
        
        # Save final model
        self.save_model()
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or train() first.")
        
        self.logger.info("Evaluating model on test data...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        from utils.metrics import MetricsCalculator
        calculator = MetricsCalculator(model_name=self.config['model']['name'])
        metrics = calculator.calculate_all_metrics(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        calculator.plot_confusion_matrix(y_test, y_pred, 
                                        title="Test Set Confusion Matrix")
        calculator.plot_roc_curve(y_test, y_pred_proba, 
                                 title="Test Set ROC Curve")
        calculator.plot_precision_recall_curve(y_test, y_pred_proba,
                                              title="Test Set Precision-Recall Curve")
        calculator.plot_metrics_comparison(metrics,
                                          title="Test Set Performance Metrics")
        
        # Generate interactive dashboard
        calculator.create_interactive_dashboard(y_test, y_pred, y_pred_proba)
        
        # Generate report
        report = calculator.generate_report(metrics, "test")
        
        self.logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def plot_training_history(self, save=True):
        """
        Plot training history
        
        Args:
            save: Whether to save the plots
            
        Returns:
            matplotlib figure
        """
        if self.history is None:
            self.logger.warning("No training history available")
            return None
        
        history = self.history.history
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flat
        
        # Plot accuracy
        if 'accuracy' in history:
            ax = axes[0]
            ax.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
            if 'val_accuracy' in history:
                ax.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot loss
        if 'loss' in history:
            ax = axes[1]
            ax.plot(history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in history:
                ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Model Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot precision
        if 'precision' in history:
            ax = axes[2]
            ax.plot(history['precision'], label='Training Precision', linewidth=2)
            if 'val_precision' in history:
                ax.plot(history['val_precision'], label='Validation Precision', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Precision')
            ax.set_title('Model Precision')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot recall
        if 'recall' in history:
            ax = axes[3]
            ax.plot(history['recall'], label='Training Recall', linewidth=2)
            if 'val_recall' in history:
                ax.plot(history['val_recall'], label='Validation Recall', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Recall')
            ax.set_title('Model Recall')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training History', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.plots_dir / 'training_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to: {save_path}")
        
        return fig
    
    def plot_model_architecture(self, save=True):
        """
        Plot model architecture
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Model architecture plot
        """
        if self.model is None:
            self.logger.warning("Model not built")
            return None
        
        try:
            plot_path = self.plots_dir / 'model_architecture.png'
            plot_model(
                self.model,
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=300
            )
            self.logger.info(f"Model architecture saved to: {plot_path}")
            return plot_path
        except ImportError:
            self.logger.warning("graphviz not installed. Skipping model architecture plot.")
            return None
    
    def save_model(self, path=None):
        """
        Save model to disk
        
        Args:
            path: Path to save model (optional)
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            self.logger.warning("No model to save")
            return None
        
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = self.config['model']['name']
            path = self.models_dir / f'{model_name}_{timestamp}_final.h5'
        
        self.model.save(path)
        self.logger.info(f"Model saved to: {path}")
        
        # Save model configuration
        config_path = path.with_suffix('.json')
        model_config = {
            'model_name': self.config['model']['name'],
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'dropout_rate': self.dropout_rate,
            'l2_regularization': self.l2_regularization,
            'batch_normalization': self.batch_normalization,
            'training_params': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer_name,
                'loss': self.loss,
                'metrics': self.metrics
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        self.logger.info(f"Model configuration saved to: {config_path}")
        
        return path
    
    def load_model(self, path):
        """
        Load model from disk
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model
        """
        if not Path(path).exists():
            self.logger.error(f"Model file not found: {path}")
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = tf.keras.models.load_model(path)
        self.logger.info(f"Model loaded from: {path}")
        
        # Try to load configuration
        config_path = path.with_suffix('.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            self.logger.info(f"Model configuration loaded from: {config_path}")
        
        return self.model
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Tuple of (probabilities, predictions)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        probabilities = self.model.predict(X, verbose=0).flatten()
        predictions = (probabilities > threshold).astype(int)
        
        return probabilities, predictions
    
    def predict_single(self, text_vector, threshold=0.5):
        """
        Make prediction for single sample
        
        Args:
            text_vector: Vectorized text
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction results
        """
        if text_vector.ndim == 1:
            text_vector = text_vector.reshape(1, -1)
        
        probability = self.model.predict(text_vector, verbose=0).flatten()[0]
        prediction = 1 if probability > threshold else 0
        
        result = {
            'probability': float(probability),
            'prediction': int(prediction),
            'class': 'SPAM' if prediction == 1 else 'HAM',
            'confidence': float(abs(probability - 0.5) * 2)  # Normalized confidence
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 1000
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    # Initialize and build model
    ann = ANNModel()
    ann.build_model()
    
    # Plot model architecture
    ann.plot_model_architecture()
    
    # Train model (with validation split)
    history = ann.train(X_train, y_train)
    
    # Plot training history
    ann.plot_training_history()
    
    # Save model
    ann.save_model()