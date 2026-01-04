"""
Optimizer and hyperparameter tuning for ANN
"""

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import optuna
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Nadam
from tensorflow.keras.callbacks import EarlyStopping

from models.ann_model import ANNModel
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger

class OptimizerTuner:
    """
    Optimizer and hyperparameter tuning for ANN model
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize OptimizerTuner
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = setup_logger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.ann_model = ANNModel(config_path)
        
        # Tuning parameters
        self.n_trials = 50
        self.cv_folds = 3
        
        # Results storage
        self.study = None
        self.trials_results = []
        self.best_params = None
        self.best_value = None
        
        # Paths
        self.plots_dir = Path(self.config['paths']['plots'])
        self.reports_dir = Path(self.config['paths']['reports'])
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Training data (will be set later)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
    
    def set_training_data(self, X_train, y_train, X_val=None, y_val=None):
        """
        Set training data for tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.logger.info(f"Training data set: {X_train.shape}")
        if X_val is not None:
            self.logger.info(f"Validation data set: {X_val.shape}")
    
    def create_optimizer(self, optimizer_name, learning_rate):
        """
        Create optimizer based on name and learning rate
        
        Args:
            optimizer_name: Name of optimizer
            learning_rate: Learning rate
            
        Returns:
            Keras optimizer
        """
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            return SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            return Adagrad(learning_rate=learning_rate)
        elif optimizer_name == 'nadam':
            return Nadam(learning_rate=learning_rate)
        else:
            self.logger.warning(f"Unknown optimizer: {optimizer_name}. Using Adam.")
            return Adam(learning_rate=learning_rate)
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy (to be maximized)
        """
        # Suggest hyperparameters
        optimizer_name = trial.suggest_categorical('optimizer', 
                                                  ['adam', 'sgd', 'rmsprop', 'nadam'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2, log=True)
        
        # Suggest hidden layer architecture
        n_layers = trial.suggest_int('n_layers', 2, 5)
        hidden_units = []
        
        # First layer units
        first_layer_units = trial.suggest_categorical('first_layer_units', 
                                                     [256, 512, 1024])
        hidden_units.append(first_layer_units)
        
        # Subsequent layers (decreasing units)
        for i in range(1, n_layers):
            reduction_factor = trial.suggest_float(f'layer_{i}_reduction', 0.3, 0.7)
            units = max(32, int(hidden_units[-1] * reduction_factor))
            hidden_units.append(units)
        
        # Create model with suggested parameters
        model = ANNModel()
        
        # Update model configuration
        model.hidden_layers = hidden_units
        model.dropout_rate = dropout_rate
        model.l2_regularization = l2_lambda
        model.learning_rate = learning_rate
        model.optimizer_name = optimizer_name
        model.batch_size = batch_size
        
        # Reduce epochs for faster tuning
        model.epochs = min(30, self.config['training']['epochs'])
        
        # Build model
        model.build_model()
        
        # Train model
        history = model.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=model.epochs,
            validation_split=0.2 if self.X_val is None else 0.0,
            validation_data=(self.X_val, self.y_val) if self.X_val is not None else None,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        # Get best validation accuracy
        best_val_accuracy = max(history.history['val_accuracy'])
        
        # Store trial results
        trial_results = {
            'trial_number': trial.number,
            'params': {
                'optimizer': optimizer_name,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate,
                'l2_lambda': l2_lambda,
                'n_layers': n_layers,
                'hidden_units': hidden_units
            },
            'best_val_accuracy': best_val_accuracy,
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
        self.trials_results.append(trial_results)
        
        return best_val_accuracy
    
    def optimize(self, n_trials=50):
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters found
        """
        self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            study_name='ann_spam_detection',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=n_trials)
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        self.logger.info(f"\nOptimization completed!")
        self.logger.info(f"Best validation accuracy: {self.best_value:.4f}")
        self.logger.info(f"Best parameters:")
        for key, value in self.best_params.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save optimization results
        self._save_optimization_results()
        
        # Create optimization plots
        self._create_optimization_plots()
        
        return self.best_params
    
    def compare_optimizers(self, optimizers=['adam', 'sgd', 'rmsprop', 'nadam'], 
                          learning_rates=[1e-4, 1e-3, 1e-2]):
        """
        Compare different optimizers and learning rates
        
        Args:
            optimizers: List of optimizers to compare
            learning_rates: List of learning rates to test
            
        Returns:
            DataFrame with comparison results
        """
        self.logger.info("Comparing optimizers and learning rates...")
        
        results = []
        
        for optimizer in optimizers:
            for lr in learning_rates:
                self.logger.info(f"Testing {optimizer} with LR={lr}")
                
                # Create and train model
                model = ANNModel()
                model.optimizer_name = optimizer
                model.learning_rate = lr
                model.epochs = 20  # Reduced for faster comparison
                
                model.build_model()
                
                start_time = time.time()
                history = model.model.fit(
                    self.X_train, self.y_train,
                    batch_size=model.batch_size,
                    epochs=model.epochs,
                    validation_split=0.2,
                    verbose=0
                )
                training_time = time.time() - start_time
                
                # Get metrics
                best_val_acc = max(history.history['val_accuracy'])
                final_val_acc = history.history['val_accuracy'][-1]
                final_val_loss = history.history['val_loss'][-1]
                
                results.append({
                    'optimizer': optimizer,
                    'learning_rate': lr,
                    'best_val_accuracy': best_val_acc,
                    'final_val_accuracy': final_val_acc,
                    'final_val_loss': final_val_loss,
                    'training_time': training_time,
                    'convergence_epoch': np.argmax(history.history['val_accuracy']) + 1
                })
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Sort by best validation accuracy
        df_results = df_results.sort_values('best_val_accuracy', ascending=False)
        
        self.logger.info("\nOptimizer Comparison Results:")
        self.logger.info(df_results.to_string())
        
        # Save results
        results_path = self.reports_dir / 'optimizer_comparison.json'
        df_results.to_json(results_path, orient='records', indent=2)
        
        # Create comparison plots
        self._plot_optimizer_comparison(df_results)
        
        return df_results
    
    def tune_architecture(self, layer_configs=None):
        """
        Tune neural network architecture
        
        Args:
            layer_configs: List of layer configurations to test
            
        Returns:
            Best architecture configuration
        """
        self.logger.info("Tuning neural network architecture...")
        
        if layer_configs is None:
            # Default configurations to test
            layer_configs = [
                [512, 256, 128, 64],    # Deep network
                [256, 128, 64],         # Medium network
                [128, 64],              # Shallow network
                [1024, 512, 256],       # Wide network
                [256, 256, 256],        # Uniform network
                [512, 128, 64, 32],     # Steep reduction
                [256, 64, 16]           # Very steep reduction
            ]
        
        results = []
        
        for i, layers in enumerate(layer_configs):
            self.logger.info(f"Testing architecture {i+1}: {layers}")
            
            # Create and train model
            model = ANNModel()
            model.hidden_layers = layers
            model.epochs = 20  # Reduced for faster testing
            
            model.build_model()
            
            history = model.model.fit(
                self.X_train, self.y_train,
                batch_size=model.batch_size,
                epochs=model.epochs,
                validation_split=0.2,
                verbose=0
            )
            
            # Get metrics
            best_val_acc = max(history.history['val_accuracy'])
            final_val_acc = history.history['val_accuracy'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            # Calculate model complexity
            total_params = model.model.count_params()
            
            results.append({
                'architecture_id': i+1,
                'layers': layers,
                'n_layers': len(layers),
                'total_neurons': sum(layers),
                'total_params': total_params,
                'best_val_accuracy': best_val_acc,
                'final_val_accuracy': final_val_acc,
                'final_val_loss': final_val_loss,
                'complexity_score': total_params / 1000  # Thousands of parameters
            })
        
        # Create DataFrame
        df_architecture = pd.DataFrame(results)
        
        # Sort by best validation accuracy
        df_architecture = df_architecture.sort_values('best_val_accuracy', ascending=False)
        
        self.logger.info("\nArchitecture Tuning Results:")
        self.logger.info(df_architecture[['architecture_id', 'layers', 'best_val_accuracy', 
                                        'total_params']].to_string())
        
        # Save results
        results_path = self.reports_dir / 'architecture_tuning.json'
        df_architecture.to_json(results_path, orient='records', indent=2)
        
        # Create architecture comparison plots
        self._plot_architecture_comparison(df_architecture)
        
        return df_architecture
    
    def _save_optimization_results(self):
        """Save optimization results to file"""
        if self.study is None:
            return
        
        # Save study results
        study_path = self.reports_dir / 'optuna_study.pkl'
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        # Save trials as JSON
        trials_path = self.reports_dir / 'optimization_trials.json'
        with open(trials_path, 'w') as f:
            json.dump(self.trials_results, f, indent=2)
        
        # Save best parameters
        best_params_path = self.reports_dir / 'best_parameters.json'
        best_params_report = {
            'best_value': self.best_value,
            'best_params': self.best_params,
            'timestamp': datetime.now().isoformat()
        }
        with open(best_params_path, 'w') as f:
            json.dump(best_params_report, f, indent=2)
        
        self.logger.info(f"Optimization results saved to: {study_path}")
        self.logger.info(f"Best parameters saved to: {best_params_path}")
    
    def _create_optimization_plots(self):
        """Create plots for optimization results"""
        if self.study is None:
            return
        
        # 1. Optimization history plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Optimization history
        ax = axes[0, 0]
        optuna.visualization.plot_optimization_history(self.study).show()
        
        # Plot 2: Parameter importances
        ax = axes[0, 1]
        try:
            optuna.visualization.plot_param_importances(self.study).show()
        except:
            ax.text(0.5, 0.5, 'Parameter importance plot\nnot available', 
                   ha='center', va='center')
            ax.set_axis_off()
        
        # Plot 3: Parallel coordinate plot
        ax = axes[1, 0]
        try:
            optuna.visualization.plot_parallel_coordinate(self.study).show()
        except:
            ax.text(0.5, 0.5, 'Parallel coordinate plot\nnot available', 
                   ha='center', va='center')
            ax.set_axis_off()
        
        # Plot 4: Slice plot
        ax = axes[1, 1]
        try:
            optuna.visualization.plot_slice(self.study).show()
        except:
            ax.text(0.5, 0.5, 'Slice plot\nnot available', 
                   ha='center', va='center')
            ax.set_axis_off()
        
        plt.suptitle('Hyperparameter Optimization Results', fontsize=16)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'optimization_results.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Optimization plots saved to: {save_path}")
    
    def _plot_optimizer_comparison(self, df_results):
        """Create plots for optimizer comparison"""
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Best validation accuracy by optimizer
        ax = axes[0, 0]
        pivot_acc = df_results.pivot_table(values='best_val_accuracy', 
                                         index='optimizer', 
                                         columns='learning_rate')
        pivot_acc.plot(kind='bar', ax=ax, colormap='viridis')
        ax.set_xlabel('Optimizer')
        ax.set_ylabel('Best Validation Accuracy')
        ax.set_title('Accuracy by Optimizer and Learning Rate')
        ax.legend(title='Learning Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Training time by optimizer
        ax = axes[0, 1]
        pivot_time = df_results.pivot_table(values='training_time', 
                                          index='optimizer', 
                                          columns='learning_rate')
        pivot_time.plot(kind='bar', ax=ax, colormap='plasma')
        ax.set_xlabel('Optimizer')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time by Optimizer and Learning Rate')
        ax.legend(title='Learning Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Convergence speed
        ax = axes[1, 0]
        pivot_conv = df_results.pivot_table(values='convergence_epoch', 
                                          index='optimizer', 
                                          columns='learning_rate')
        pivot_conv.plot(kind='bar', ax=ax, colormap='coolwarm')
        ax.set_xlabel('Optimizer')
        ax.set_ylabel('Convergence Epoch')
        ax.set_title('Convergence Speed by Optimizer and Learning Rate')
        ax.legend(title='Learning Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Best overall performers
        ax = axes[1, 1]
        top_n = min(10, len(df_results))
        top_results = df_results.head(top_n).copy()
        top_results['config'] = top_results['optimizer'] + '_LR' + \
                               top_results['learning_rate'].astype(str)
        
        x = np.arange(top_n)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, top_results['best_val_accuracy'], 
                      width, label='Accuracy', color='skyblue')
        bars2 = ax.bar(x + width/2, top_results['final_val_loss'], 
                      width, label='Loss', color='lightcoral')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Score')
        ax.set_title(f'Top {top_n} Best Performing Configurations')
        ax.set_xticks(x)
        ax.set_xticklabels(top_results['config'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Optimizer and Learning Rate Comparison', fontsize=16)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'optimizer_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Optimizer comparison plots saved to: {save_path}")
    
    def _plot_architecture_comparison(self, df_architecture):
        """Create plots for architecture comparison"""
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Accuracy vs Complexity
        ax = axes[0, 0]
        scatter = ax.scatter(df_architecture['complexity_score'], 
                           df_architecture['best_val_accuracy'],
                           c=df_architecture['n_layers'],
                           s=df_architecture['total_neurons']/10,
                           cmap='viridis',
                           alpha=0.7,
                           edgecolors='black')
        
        ax.set_xlabel('Model Complexity (Thousands of Parameters)')
        ax.set_ylabel('Best Validation Accuracy')
        ax.set_title('Accuracy vs Model Complexity')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for number of layers
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Layers')
        
        # Annotate points with architecture IDs
        for idx, row in df_architecture.iterrows():
            ax.annotate(str(row['architecture_id']),
                       (row['complexity_score'], row['best_val_accuracy']),
                       fontsize=8, ha='center', va='center')
        
        # Plot 2: Accuracy by Number of Layers
        ax = axes[0, 1]
        df_grouped = df_architecture.groupby('n_layers')['best_val_accuracy'].agg(['mean', 'std'])
        bars = ax.bar(df_grouped.index, df_grouped['mean'], 
                     yerr=df_grouped['std'],
                     capsize=5, color='lightgreen', edgecolor='darkgreen')
        
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Average Validation Accuracy')
        ax.set_title('Accuracy by Number of Layers')
        ax.set_xticks(df_grouped.index)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean_val in zip(bars, df_grouped['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom')
        
        # Plot 3: Top performing architectures
        ax = axes[1, 0]
        top_n = min(8, len(df_architecture))
        top_arch = df_architecture.head(top_n).copy()
        
        x = np.arange(top_n)
        width = 0.25
        
        bars1 = ax.bar(x - width, top_arch['best_val_accuracy'], 
                      width, label='Accuracy', color='skyblue')
        bars2 = ax.bar(x, top_arch['final_val_loss'], 
                      width, label='Loss', color='lightcoral')
        bars3 = ax.bar(x + width, top_arch['complexity_score'], 
                      width, label='Complexity', color='lightgreen')
        
        ax.set_xlabel('Architecture ID')
        ax.set_ylabel('Score')
        ax.set_title(f'Top {top_n} Architectures Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(top_arch['architecture_id'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Layer size patterns
        ax = axes[1, 1]
        for idx, row in df_architecture.head(6).iterrows():
            layers = row['layers']
            ax.plot(range(1, len(layers) + 1), layers, 
                   marker='o', label=f"Arch {row['architecture_id']}")
        
        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Number of Neurons')
        ax.set_title('Layer Size Patterns')
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Neural Network Architecture Analysis', fontsize=16)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'architecture_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Architecture comparison plots saved to: {save_path}")
    
    def get_recommended_configuration(self):
        """
        Get recommended configuration based on optimization results
        
        Returns:
            Dictionary with recommended configuration
        """
        if self.best_params is None:
            self.logger.warning("No optimization results available. Running optimization...")
            self.optimize(n_trials=20)
        
        recommendation = {
            'recommended_optimizer': self.best_params.get('optimizer', 'adam'),
            'recommended_learning_rate': self.best_params.get('learning_rate', 0.001),
            'recommended_batch_size': self.best_params.get('batch_size', 32),
            'recommended_dropout_rate': self.best_params.get('dropout_rate', 0.3),
            'recommended_l2_lambda': self.best_params.get('l2_lambda', 0.001),
            'recommended_architecture': {
                'n_layers': self.best_params.get('n_layers', 3),
                'hidden_units': self._get_hidden_units_from_params(self.best_params)
            },
            'expected_accuracy': self.best_value if self.best_value else 0.85,
            'optimization_method': 'Optuna Bayesian Optimization',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save recommendation
        rec_path = self.reports_dir / 'recommended_configuration.json'
        with open(rec_path, 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        self.logger.info(f"Recommended configuration saved to: {rec_path}")
        
        return recommendation
    
    def _get_hidden_units_from_params(self, params):
        """
        Extract hidden units from parameters
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            List of hidden units
        """
        hidden_units = []
        
        if 'hidden_units' in params:
            return params['hidden_units']
        
        # Reconstruct from individual parameters
        n_layers = params.get('n_layers', 3)
        first_layer_units = params.get('first_layer_units', 256)
        
        hidden_units.append(first_layer_units)
        
        for i in range(1, n_layers):
            reduction_key = f'layer_{i}_reduction'
            if reduction_key in params:
                units = max(32, int(hidden_units[-1] * params[reduction_key]))
            else:
                units = max(32, int(hidden_units[-1] * 0.5))  # Default 50% reduction
            hidden_units.append(units)
        
        return hidden_units

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 500
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    # Initialize tuner
    tuner = OptimizerTuner()
    tuner.set_training_data(X_train, y_train)
    
    # Compare optimizers
    df_opt_comparison = tuner.compare_optimizers()
    
    # Tune architecture
    df_arch_tuning = tuner.tune_architecture()
    
    # Run full optimization
    best_params = tuner.optimize(n_trials=30)
    
    # Get recommended configuration
    recommendation = tuner.get_recommended_configuration()
    
    print("\nOptimization completed!")
    print(f"Best validation accuracy: {tuner.best_value:.4f}")
    print(f"Recommended configuration:")
    for key, value in recommendation.items():
        if key != 'recommended_architecture':
            print(f"  {key}: {value}")