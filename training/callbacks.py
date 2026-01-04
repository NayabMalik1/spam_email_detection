"""
Custom callbacks for ANN training
"""

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import logging

class TrainingMonitor(Callback):
    """
    Custom callback for monitoring training progress
    """
    
    def __init__(self, model_name="ANN_Model", save_dir="outputs/plots", 
                 log_frequency=5):
        """
        Initialize TrainingMonitor
        
        Args:
            model_name: Name of the model
            save_dir: Directory to save plots
            log_frequency: Frequency of logging (epochs)
        """
        super().__init__()
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_frequency = log_frequency
        
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Best metrics
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        # Training start time
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.logger.info(f"Starting training for {self.model_name}")
        self.start_time = datetime.now()
        
        # Initialize plots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.axes = self.axes.flat
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        logs = logs or {}
        
        # Update history
        self.history['epoch'].append(epoch)
        for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
            if key in logs:
                self.history[key].append(logs[key])
        
        # Get current learning rate
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.history['learning_rate'].append(float(lr))
        
        # Update best metrics
        if 'val_accuracy' in logs and logs['val_accuracy'] > self.best_accuracy:
            self.best_accuracy = logs['val_accuracy']
            self.best_loss = logs.get('val_loss', float('inf'))
            self.best_epoch = epoch
        
        # Log progress
        if epoch % self.log_frequency == 0:
            elapsed = datetime.now() - self.start_time
            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"Loss: {logs.get('loss', 0):.4f} | "
                f"Acc: {logs.get('accuracy', 0):.4f} | "
                f"Val Loss: {logs.get('val_loss', 0):.4f} | "
                f"Val Acc: {logs.get('val_accuracy', 0):.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {elapsed}"
            )
        
        # Update plots every log_frequency epochs
        if epoch % self.log_frequency == 0:
            self._update_plots(epoch)
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        elapsed = datetime.now() - self.start_time
        
        self.logger.info(f"\nTraining completed!")
        self.logger.info(f"Total training time: {elapsed}")
        self.logger.info(f"Best validation accuracy: {self.best_accuracy:.4f} at epoch {self.best_epoch}")
        self.logger.info(f"Best validation loss: {self.best_loss:.4f}")
        
        # Save final plots
        self._save_final_plots()
        
        # Save training history
        self._save_history()
    
    def _update_plots(self, epoch):
        """Update training plots"""
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Plot 1: Loss
        ax = self.axes[0]
        if len(self.history['loss']) > 0:
            ax.plot(self.history['epoch'], self.history['loss'], 
                   label='Training Loss', linewidth=2)
        if len(self.history['val_loss']) > 0:
            ax.plot(self.history['epoch'], self.history['val_loss'], 
                   label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax = self.axes[1]
        if len(self.history['accuracy']) > 0:
            ax.plot(self.history['epoch'], self.history['accuracy'], 
                   label='Training Accuracy', linewidth=2)
        if len(self.history['val_accuracy']) > 0:
            ax.plot(self.history['epoch'], self.history['val_accuracy'], 
                   label='Validation Accuracy', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate
        ax = self.axes[2]
        if len(self.history['learning_rate']) > 0:
            ax.plot(self.history['epoch'], self.history['learning_rate'], 
                   linewidth=2, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss vs Accuracy
        ax = self.axes[3]
        if len(self.history['loss']) > 0 and len(self.history['accuracy']) > 0:
            scatter = ax.scatter(self.history['loss'], self.history['accuracy'],
                               c=self.history['epoch'], cmap='viridis',
                               s=50, alpha=0.6)
            ax.set_xlabel('Loss')
            ax.set_ylabel('Accuracy')
            ax.set_title('Loss vs Accuracy Progression')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Epoch')
        
        plt.suptitle(f'{self.model_name} - Training Progress (Epoch {epoch})', 
                    fontsize=14)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def _save_final_plots(self):
        """Save final training plots"""
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flat
        
        # Plot 1: Loss
        ax = axes[0]
        if len(self.history['loss']) > 0:
            ax.plot(self.history['epoch'], self.history['loss'], 
                   label='Training Loss', linewidth=2)
        if len(self.history['val_loss']) > 0:
            ax.plot(self.history['epoch'], self.history['val_loss'], 
                   label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax = axes[1]
        if len(self.history['accuracy']) > 0:
            ax.plot(self.history['epoch'], self.history['accuracy'], 
                   label='Training Accuracy', linewidth=2)
        if len(self.history['val_accuracy']) > 0:
            ax.plot(self.history['epoch'], self.history['val_accuracy'], 
                   label='Validation Accuracy', linewidth=2)
        
        # Mark best accuracy
        if self.best_accuracy > 0:
            ax.axhline(y=self.best_accuracy, color='red', linestyle='--', 
                      alpha=0.5, label=f'Best: {self.best_accuracy:.4f}')
            ax.axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate
        ax = axes[2]
        if len(self.history['learning_rate']) > 0:
            ax.plot(self.history['epoch'], self.history['learning_rate'], 
                   linewidth=2, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss vs Accuracy
        ax = axes[3]
        if len(self.history['loss']) > 0 and len(self.history['accuracy']) > 0:
            scatter = ax.scatter(self.history['loss'], self.history['accuracy'],
                               c=self.history['epoch'], cmap='viridis',
                               s=50, alpha=0.6)
            ax.set_xlabel('Loss')
            ax.set_ylabel('Accuracy')
            ax.set_title('Loss vs Accuracy')
            ax.grid(True, alpha=0.3)
            
            # Mark best point
            if self.best_epoch < len(self.history['loss']):
                ax.scatter(self.history['loss'][self.best_epoch],
                          self.history['accuracy'][self.best_epoch],
                          s=200, color='red', marker='*', 
                          label=f'Best (Epoch {self.best_epoch})')
                ax.legend()
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Epoch')
        
        # Plot 5: Convergence
        ax = axes[4]
        if len(self.history['loss']) > 0:
            # Calculate convergence metric (rate of loss decrease)
            loss_diff = np.diff(self.history['loss'])
            convergence_rate = -loss_diff  # Negative of difference (decrease is positive)
            
            if len(convergence_rate) > 0:
                ax.plot(self.history['epoch'][1:], convergence_rate, 
                       linewidth=2, color='green')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss Decrease Rate')
                ax.set_title('Convergence Rate')
                ax.grid(True, alpha=0.3)
        
        # Plot 6: Summary statistics
        ax = axes[5]
        ax.axis('off')
        
        summary_text = f"""
        {self.model_name}
        
        Training Statistics:
        --------------------
        Total Epochs: {len(self.history['epoch'])}
        Best Epoch: {self.best_epoch}
        Best Accuracy: {self.best_accuracy:.4f}
        Best Loss: {self.best_loss:.4f}
        
        Final Metrics:
        ---------------
        Final Loss: {self.history['loss'][-1] if self.history['loss'] else 0:.4f}
        Final Accuracy: {self.history['accuracy'][-1] if self.history['accuracy'] else 0:.4f}
        Final Val Loss: {self.history['val_loss'][-1] if self.history['val_loss'] else 0:.4f}
        Final Val Accuracy: {self.history['val_accuracy'][-1] if self.history['val_accuracy'] else 0:.4f}
        Final Learning Rate: {self.history['learning_rate'][-1] if self.history['learning_rate'] else 0:.6f}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{self.model_name} - Training Summary', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / f'{self.model_name}_training_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training summary plot saved to: {save_path}")
    
    def _save_history(self):
        """Save training history to file"""
        history_path = self.save_dir / f'{self.model_name}_training_history.json'
        
        history_data = {
            'model_name': self.model_name,
            'training_start': self.start_time.isoformat() if self.start_time else None,
            'training_end': datetime.now().isoformat(),
            'best_epoch': self.best_epoch,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'history': self.history
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        self.logger.info(f"Training history saved to: {history_path}")

class GradientMonitor(Callback):
    """
    Monitor gradients during training to detect vanishing/exploding gradients
    """
    
    def __init__(self, model_name="ANN_Model", save_dir="outputs/plots"):
        """
        Initialize GradientMonitor
        
        Args:
            model_name: Name of the model
            save_dir: Directory to save plots
        """
        super().__init__()
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Gradient history
        self.gradient_history = {
            'epoch': [],
            'layer_gradients': {}
        }
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        gradients = []
        layer_names = []
        
        # Get gradients for each trainable variable
        for layer in self.model.layers:
            if layer.trainable_weights:
                for weight in layer.trainable_weights:
                    grad = self.model.optimizer.get_gradients(
                        self.model.total_loss, weight
                    )
                    if grad is not None:
                        grad_value = tf.keras.backend.get_value(grad)
                        if grad_value is not None:
                            grad_norm = np.linalg.norm(grad_value)
                            gradients.append(grad_norm)
                            layer_names.append(f"{layer.name}_{weight.name.split('/')[-1]}")
        
        # Store gradient statistics
        if gradients:
            self.gradient_history['epoch'].append(epoch)
            
            if epoch == 0:
                # Initialize layer gradients dictionary
                for name in layer_names:
                    self.gradient_history['layer_gradients'][name] = []
            
            # Store gradients for each layer
            for name, grad in zip(layer_names, gradients):
                if name in self.gradient_history['layer_gradients']:
                    self.gradient_history['layer_gradients'][name].append(grad)
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        if not self.gradient_history['epoch']:
            return
        
        # Create gradient analysis plots
        self._plot_gradient_analysis()
        
        # Save gradient history
        self._save_gradient_history()
    
    def _plot_gradient_analysis(self):
        """Create gradient analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flat
        
        # Plot 1: Gradient norms over epochs
        ax = axes[0]
        for layer_name, gradients in self.gradient_history['layer_gradients'].items():
            if len(gradients) > 0:
                ax.plot(self.gradient_history['epoch'], gradients, 
                       label=layer_name, alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms by Layer')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 2: Gradient distribution histogram
        ax = axes[1]
        all_gradients = []
        for gradients in self.gradient_history['layer_gradients'].values():
            all_gradients.extend(gradients)
        
        if all_gradients:
            ax.hist(all_gradients, bins=50, alpha=0.7, color='skyblue', 
                   edgecolor='black', log=True)
            ax.set_xlabel('Gradient Norm')
            ax.set_ylabel('Frequency (log)')
            ax.set_title('Gradient Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_grad = np.mean(all_gradients)
            std_grad = np.std(all_gradients)
            ax.axvline(x=mean_grad, color='red', linestyle='--', 
                      label=f'Mean: {mean_grad:.2e}')
            ax.axvline(x=mean_grad + std_grad, color='orange', 
                      linestyle=':', label=f'±1 Std')
            ax.axvline(x=mean_grad - std_grad, color='orange', 
                      linestyle=':')
            ax.legend()
        
        # Plot 3: Gradient statistics by layer (final epoch)
        ax = axes[2]
        if self.gradient_history['epoch']:
            final_epoch = self.gradient_history['epoch'][-1]
            layer_names = []
            final_gradients = []
            
            for layer_name, gradients in self.gradient_history['layer_gradients'].items():
                if gradients:
                    layer_names.append(layer_name)
                    final_gradients.append(gradients[-1])
            
            if final_gradients:
                bars = ax.barh(range(len(layer_names)), final_gradients, 
                              color=plt.cm.Set3(np.arange(len(layer_names))))
                ax.set_yticks(range(len(layer_names)))
                ax.set_yticklabels(layer_names, fontsize=8)
                ax.set_xlabel('Gradient Norm (Final Epoch)')
                ax.set_title(f'Final Gradient Norms by Layer (Epoch {final_epoch})')
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Gradient issues detection
        ax = axes[3]
        if all_gradients:
            # Detect vanishing gradients (norm < 1e-7)
            vanishing_threshold = 1e-7
            vanishing_counts = []
            
            # Detect exploding gradients (norm > 1.0)
            exploding_threshold = 1.0
            exploding_counts = []
            
            for epoch_idx, epoch in enumerate(self.gradient_history['epoch']):
                epoch_gradients = []
                for gradients in self.gradient_history['layer_gradients'].values():
                    if epoch_idx < len(gradients):
                        epoch_gradients.append(gradients[epoch_idx])
                
                if epoch_gradients:
                    vanishing_count = sum(1 for g in epoch_gradients 
                                        if g < vanishing_threshold)
                    exploding_count = sum(1 for g in epoch_gradients 
                                        if g > exploding_threshold)
                    
                    vanishing_counts.append(vanishing_count)
                    exploding_counts.append(exploding_count)
            
            if vanishing_counts and exploding_counts:
                ax.plot(self.gradient_history['epoch'], vanishing_counts, 
                       label='Vanishing Gradients', color='red', linewidth=2)
                ax.plot(self.gradient_history['epoch'], exploding_counts, 
                       label='Exploding Gradients', color='blue', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Number of Layers')
                ax.set_title('Gradient Issues Detection')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.model_name} - Gradient Analysis', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / f'{self.model_name}_gradient_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_gradient_history(self):
        """Save gradient history to file"""
        grad_path = self.save_dir / f'{self.model_name}_gradient_history.json'
        
        grad_data = {
            'model_name': self.model_name,
            'gradient_history': self.gradient_history
        }
        
        with open(grad_path, 'w') as f:
            json.dump(grad_data, f, indent=2, default=lambda x: float(x) 
                     if isinstance(x, (np.float32, np.float64)) else x)
        
        logging.getLogger(__name__).info(f"Gradient history saved to: {grad_path}")

class PredictionSaver(Callback):
    """
    Save model predictions during training for analysis
    """
    
    def __init__(self, X_val, y_val, model_name="ANN_Model", 
                 save_dir="outputs/reports", save_frequency=10):
        """
        Initialize PredictionSaver
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            model_name: Name of the model
            save_dir: Directory to save predictions
            save_frequency: Frequency of saving predictions (epochs)
        """
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        
        # Prediction history
        self.prediction_history = {
            'epoch': [],
            'predictions': [],
            'probabilities': [],
            'metrics': []
        }
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        if epoch % self.save_frequency == 0:
            # Get predictions
            y_pred_proba = self.model.predict(self.X_val, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from utils.metrics import calculate_all_metrics
            metrics = calculate_all_metrics(self.y_val, y_pred, y_pred_proba)
            
            # Store predictions and metrics
            self.prediction_history['epoch'].append(epoch)
            self.prediction_history['predictions'].append(y_pred.tolist())
            self.prediction_history['probabilities'].append(y_pred_proba.tolist())
            self.prediction_history['metrics'].append(metrics)
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        # Save prediction history
        self._save_prediction_history()
        
        # Create prediction analysis plots
        self._plot_prediction_analysis()
    
    def _save_prediction_history(self):
        """Save prediction history to file"""
        pred_path = self.save_dir / f'{self.model_name}_prediction_history.json'
        
        # Convert numpy arrays to lists for JSON serialization
        pred_data = {
            'model_name': self.model_name,
            'X_val_shape': list(self.X_val.shape) if hasattr(self.X_val, 'shape') else None,
            'y_val_shape': list(self.y_val.shape) if hasattr(self.y_val, 'shape') else None,
            'prediction_history': self.prediction_history
        }
        
        with open(pred_path, 'w') as f:
            json.dump(pred_data, f, indent=2, default=lambda x: float(x) 
                     if isinstance(x, (np.float32, np.float64)) else x)
        
        logging.getLogger(__name__).info(f"Prediction history saved to: {pred_path}")
    
    def _plot_prediction_analysis(self):
        """Create prediction analysis plots"""
        if not self.prediction_history['epoch']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flat
        
        # Plot 1: Accuracy over epochs
        ax = axes[0]
        epochs = self.prediction_history['epoch']
        accuracies = [metrics['accuracy'] for metrics in self.prediction_history['metrics']]
        
        ax.plot(epochs, accuracies, 'o-', linewidth=2, markersize=8, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy During Training')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Probability distributions
        ax = axes[1]
        if self.prediction_history['probabilities']:
            # Get final epoch probabilities
            final_probs = self.prediction_history['probabilities'][-1]
            
            # Separate by true class
            ham_probs = [p for p, y in zip(final_probs, self.y_val) if y == 0]
            spam_probs = [p for p, y in zip(final_probs, self.y_val) if y == 1]
            
            ax.hist(ham_probs, bins=30, alpha=0.5, label='Ham', 
                   color='green', edgecolor='black')
            ax.hist(spam_probs, bins=30, alpha=0.5, label='Spam', 
                   color='red', edgecolor='black')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Frequency')
            ax.set_title('Prediction Probability Distribution (Final Epoch)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Confidence evolution
        ax = axes[2]
        confidence_by_epoch = []
        
        for probs in self.prediction_history['probabilities']:
            # Calculate average confidence (distance from 0.5)
            confidences = [abs(p - 0.5) * 2 for p in probs]  # Normalize to [0, 1]
            avg_confidence = np.mean(confidences)
            confidence_by_epoch.append(avg_confidence)
        
        ax.plot(epochs, confidence_by_epoch, 's-', linewidth=2, 
               markersize=8, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Prediction Confidence During Training')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Error analysis (final epoch)
        ax = axes[3]
        if self.prediction_history['predictions']:
            final_preds = self.prediction_history['predictions'][-1]
            
            # Calculate errors
            errors = np.array(final_preds) != np.array(self.y_val)
            error_indices = np.where(errors)[0]
            
            if len(error_indices) > 0:
                # Get error probabilities
                error_probs = [final_probs[i] for i in error_indices]
                error_types = ['FP' if pred == 1 and true == 0 else 'FN' 
                              for pred, true in zip(final_preds, self.y_val) 
                              if pred != true]
                
                # Create error analysis
                fp_count = error_types.count('FP')
                fn_count = error_types.count('FN')
                
                bars = ax.bar(['False Positives', 'False Negatives'], 
                             [fp_count, fn_count], 
                             color=['orange', 'red'])
                ax.set_ylabel('Count')
                ax.set_title(f'Error Analysis (Final Epoch)\nTotal Errors: {len(error_indices)}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, count in zip(bars, [fp_count, fn_count]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           str(count), ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No Errors in Final Epoch!', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, fontweight='bold')
                ax.set_title('Error Analysis (Final Epoch)')
        
        plt.suptitle(f'{self.model_name} - Prediction Analysis', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / f'{self.model_name}_prediction_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def get_callbacks(config, X_val=None, y_val=None):
    """
    Get list of callbacks based on configuration
    
    Args:
        config: Configuration dictionary
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    model_name = config['model']['name']
    save_dir = config['paths']['plots']
    
    # Training monitor
    callbacks.append(TrainingMonitor(
        model_name=model_name,
        save_dir=save_dir,
        log_frequency=5
    ))
    
    # Gradient monitor
    callbacks.append(GradientMonitor(
        model_name=model_name,
        save_dir=save_dir
    ))
    
    # Prediction saver (if validation data provided)
    if X_val is not None and y_val is not None:
        callbacks.append(PredictionSaver(
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
            save_dir=config['paths']['reports'],
            save_frequency=10
        ))
    
    # Early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    ))
    
    # Reduce learning rate on plateau
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['training']['reduce_lr_factor'],
        patience=config['training']['reduce_lr_patience'],
        min_lr=1e-6,
        verbose=1
    ))
    
    # Model checkpoint
    from tensorflow.keras.callbacks import ModelCheckpoint
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = Path(config['paths']['models']) / f'{model_name}_{timestamp}.h5'
    
    callbacks.append(ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ))
    
    # TensorBoard
    from tensorflow.keras.callbacks import TensorBoard
    
    log_dir = Path(config['paths']['logs']) / f'tensorboard_{timestamp}'
    callbacks.append(TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    ))
    
    # CSV logger
    from tensorflow.keras.callbacks import CSVLogger
    
    csv_path = Path(config['paths']['logs']) / f'training_{timestamp}.csv'
    callbacks.append(CSVLogger(
        filename=csv_path,
        separator=',',
        append=False
    ))
    
    return callbacks

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 100)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 100)
    y_val = np.random.randint(0, 2, 20)
    
    # Create simple model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(100,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create callbacks
    config = {
        'model': {'name': 'Test_Model'},
        'paths': {
            'plots': 'outputs/plots',
            'reports': 'outputs/reports',
            'models': 'models/saved_models',
            'logs': 'logs'
        },
        'training': {
            'early_stopping_patience': 10,
            'reduce_lr_factor': 0.5,
            'reduce_lr_patience': 5
        }
    }
    
    callbacks_list = get_callbacks(config, X_val, y_val)
    
    # Train model with callbacks
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("Training completed with custom callbacks!")