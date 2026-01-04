"""
Training pipeline for ANN spam detection model
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
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Any
import pickle

from models.ann_model import ANNModel
from models.text_vectorizer import TextVectorizer
from utils.text_cleaner import TextCleaner
from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger, log_performance_metrics

class ModelTrainer:
    """
    Complete training pipeline for spam detection ANN
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize ModelTrainer
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = setup_logger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.vectorizer = TextVectorizer(config_path)
        self.ann_model = ANNModel(config_path)
        self.data_loader = DataLoader(config_path)
        self.metrics_calculator = MetricsCalculator()
        
        # Training parameters
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        
        # Paths
        self.models_dir = Path(self.config['paths']['models'])
        self.plots_dir = Path(self.config['paths']['plots'])
        self.reports_dir = Path(self.config['paths']['reports'])
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Training data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Results
        self.training_history = None
        self.evaluation_results = None
        self.cross_val_results = None
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess data
        
        Returns:
            Dictionary with data splits
        """
        self.logger.info("Loading and preprocessing data...")
        
        # Load splits
        splits = self.data_loader.load_splits()
        
        # Get processed directory from data_loader
        processed_dir = self.data_loader.processed_dir
        
        # Check if cleaned data already exists
        cleaned_data_path = processed_dir / 'cleaned_data.pkl'
        
        if cleaned_data_path.exists():
            self.logger.info("Loading cached cleaned data...")
            with open(cleaned_data_path, 'rb') as f:
                splits = pickle.load(f)
            return splits
        
        # If not cached, clean the data
        all_labels = []
        
        for split_name, split_df in splits.items():
            if split_df is None:
                continue
                
            self.logger.info(f"Cleaning text in {split_name} set...")
            
            # Clean text
            splits[split_name]['cleaned_text'] = self.text_cleaner.clean_batch(
                split_df[self.config['data']['text_column']].tolist(),
                show_progress=True
            )
            
            # Store labels
            all_labels.extend(split_df[self.config['data']['label_column']].tolist())
        
        # Save cleaned data for future use
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open(cleaned_data_path, 'wb') as f:
            pickle.dump(splits, f)
        
        self.logger.info(f"Cleaned data saved to: {cleaned_data_path}")
        
        return splits
    
    def vectorize_data(self, splits):
        """
        Vectorize text data
        
        Args:
            splits: Dictionary with data splits
            
        Returns:
            Dictionary with vectorized data
        """
        self.logger.info("Vectorizing text data...")
        
        # Prepare all text for fitting vectorizer
        all_texts = []
        for split_name, split_df in splits.items():
            all_texts.extend(split_df['cleaned_text'].tolist())
        
        # Fit vectorizer on all data
        self.logger.info("Fitting vectorizer...")
        X_all = self.vectorizer.fit_transform(all_texts, method='tfidf')
        
        # Split back into train/val/test
        split_indices = {
            'train': (0, len(splits['train'])),
            'val': (len(splits['train']), len(splits['train']) + len(splits['val'])),
            'test': (len(splits['train']) + len(splits['val']), len(all_texts))
        }
        
        vectorized_data = {}
        for split_name, (start, end) in split_indices.items():
            X = X_all[start:end]
            y = splits[split_name][self.config['data']['label_column']].values
            
            vectorized_data[split_name] = {
                'X': X,
                'y': y,
                'texts': splits[split_name]['cleaned_text'].tolist(),
                'original_texts': splits[split_name][self.config['data']['text_column']].tolist()
            }
            
            self.logger.info(f"{split_name.upper()} set shape: {X.shape}")
        
        # Skip feature distribution plot to save memory
        self.logger.info("Skipped feature distribution plot to save memory")

        # Save vectorizer
        self.vectorizer.save_vectorizer()
        
        # Store vectorized data
        self.vectorized_data = vectorized_data
        
        # Assign to instance variables
        self.X_train = vectorized_data['train']['X']
        self.y_train = vectorized_data['train']['y']
        self.X_val = vectorized_data['val']['X']
        self.y_val = vectorized_data['val']['y']
        self.X_test = vectorized_data['test']['X']
        self.y_test = vectorized_data['test']['y']
        
        return vectorized_data
    
    def train_model(self):
        """
        Train the ANN model
        
        Returns:
            Training history
        """
        self.logger.info("Starting model training...")
        
        try:
            # Build model
            self.ann_model.build_model()
            
            # Plot model architecture
            self.ann_model.plot_model_architecture()
            
            # Train model
            start_time = time.time()
            
            self.training_history = self.ann_model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val
            )
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Plot training history
            self.ann_model.plot_training_history()
            
            # Save model
            self.ann_model.save_model()
            
            return self.training_history
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}", exc_info=True)
            raise
    
    def evaluate_model(self):
        """
        Evaluate model performance
        
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Evaluating model...")
        
        try:
            # Evaluate on test set
            test_metrics = self.ann_model.evaluate(self.X_test, self.y_test)
            
            # Evaluate on validation set
            val_pred_proba = self.ann_model.model.predict(self.X_val, verbose=0).flatten()
            val_pred = (val_pred_proba > 0.5).astype(int)
            
            from utils.metrics import MetricsCalculator
            val_calculator = MetricsCalculator(model_name=self.config['model']['name'])
            val_metrics = val_calculator.calculate_all_metrics(
                self.y_val, val_pred, val_pred_proba
            )
            
            # Store results
            self.evaluation_results = {
                'test': test_metrics,
                'validation': val_metrics,
                'timestamp': datetime.now().isoformat(),
                'model_config': {
                    'input_dim': self.ann_model.input_dim,
                    'hidden_layers': self.ann_model.hidden_layers,
                    'activation': self.ann_model.activation,
                    'dropout_rate': self.ann_model.dropout_rate
                }
            }
            
            # Log performance metrics
            self.logger.info("\n" + "="*60)
            self.logger.info("MODEL EVALUATION RESULTS")
            self.logger.info("="*60)
            
            for dataset in ['validation', 'test']:
                metrics = self.evaluation_results[dataset]
                self.logger.info(f"\n{dataset.upper()} SET:")
                self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                self.logger.info(f"  Precision: {metrics['precision']:.4f}")
                self.logger.info(f"  Recall: {metrics['recall']:.4f}")
                self.logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
                if 'roc_auc' in metrics:
                    self.logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Save evaluation results
            self._save_evaluation_results()
            
            # Create comparison plot
            self._plot_metrics_comparison()
            
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
            raise
    
    def cross_validate(self, n_splits=5):
        """
        Perform cross-validation
        
        Args:
            n_splits: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        try:
            # Combine train and validation for CV
            X_cv = np.vstack([self.X_train, self.X_val])
            y_cv = np.concatenate([self.y_train, self.y_val])
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            cv_results = {
                'fold_metrics': [],
                'mean_metrics': {},
                'std_metrics': {}
            }
            
            fold_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'roc_auc': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv), 1):
                self.logger.info(f"\nTraining Fold {fold}/{n_splits}")
                
                # Split data for this fold
                X_train_fold = X_cv[train_idx]
                y_train_fold = y_cv[train_idx]
                X_val_fold = X_cv[val_idx]
                y_val_fold = y_cv[val_idx]
                
                # Create and train model for this fold
                fold_model = ANNModel()
                fold_model.build_model()
                
                # Train with reduced epochs for CV
                fold_model.epochs = min(20, self.epochs)
                fold_model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                
                # Evaluate on validation fold
                y_pred_proba = fold_model.model.predict(X_val_fold, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate metrics
                from utils.metrics import MetricsCalculator
                calculator = MetricsCalculator()
                metrics = calculator.calculate_all_metrics(y_val_fold, y_pred, y_pred_proba)
                
                # Store fold results
                cv_results['fold_metrics'].append({
                    'fold': fold,
                    'metrics': metrics
                })
                
                # Collect scores
                for metric in fold_scores.keys():
                    if metric in metrics:
                        fold_scores[metric].append(metrics[metric])
                
                self.logger.info(f"Fold {fold} - Accuracy: {metrics['accuracy']:.4f}, "
                              f"F1-Score: {metrics['f1_score']:.4f}")
            
            # Calculate mean and std
            for metric, scores in fold_scores.items():
                if scores:
                    cv_results['mean_metrics'][metric] = np.mean(scores)
                    cv_results['std_metrics'][metric] = np.std(scores)
            
            self.cross_val_results = cv_results
            
            # Log CV results
            self.logger.info("\n" + "="*60)
            self.logger.info("CROSS-VALIDATION RESULTS")
            self.logger.info("="*60)
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if metric in cv_results['mean_metrics']:
                    mean_val = cv_results['mean_metrics'][metric]
                    std_val = cv_results['std_metrics'][metric]
                    self.logger.info(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Save CV results
            self._save_cv_results()
            
            # Plot CV results
            self._plot_cv_results()
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}", exc_info=True)
            raise
    
    def run_full_pipeline(self):
        """
        Run complete training pipeline
        
        Returns:
            Dictionary with all results
        """
        self.logger.info("="*60)
        self.logger.info("STARTING FULL TRAINING PIPELINE")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 1. Load and preprocess data
            splits = self.load_and_preprocess_data()
            
            # 2. Vectorize data
            vectorized_data = self.vectorize_data(splits)
            
            # 3. Train model
            history = self.train_model()
            
            # 4. Evaluate model
            evaluation_results = self.evaluate_model()
            
            # 5. Cross-validation (optional)
            cv_results = self.cross_validate(n_splits=5)
            
            total_time = time.time() - start_time
            
            # Create final report
            final_report = self._create_final_report(
                splits, vectorized_data, history, 
                evaluation_results, cv_results, total_time
            )
            
            self.logger.info("\n" + "="*60)
            self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total time: {total_time:.2f} seconds")
            self.logger.info("="*60)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def _save_evaluation_results(self):
        """Save evaluation results to file"""
        if self.evaluation_results:
            file_path = self.reports_dir / 'evaluation_results.json'
            with open(file_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)
            self.logger.info(f"Evaluation results saved to: {file_path}")
    
    def _save_cv_results(self):
        """Save cross-validation results to file"""
        if self.cross_val_results:
            file_path = self.reports_dir / 'cross_validation_results.json'
            with open(file_path, 'w') as f:
                json.dump(self.cross_val_results, f, indent=2)
            self.logger.info(f"Cross-validation results saved to: {file_path}")
    
    def _plot_metrics_comparison(self):
        """Plot comparison of metrics across datasets"""
        if not self.evaluation_results:
            return
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        datasets = ['validation', 'test']
        metric_values = {}
        
        for metric in metrics_to_plot:
            values = []
            for dataset in datasets:
                if metric in self.evaluation_results[dataset]:
                    values.append(self.evaluation_results[dataset][metric])
                else:
                    values.append(0)
            metric_values[metric] = values
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, (metric, values) in enumerate(metric_values.items()):
            offset = (i - len(metrics_to_plot)/2) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = self.plots_dir / 'performance_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Performance comparison plot saved to: {save_path}")
    
    def _plot_cv_results(self):
        """Plot cross-validation results"""
        if not self.cross_val_results:
            return
        
        fold_metrics = self.cross_val_results['fold_metrics']
        metrics_to_plot = ['accuracy', 'f1_score']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            fold_numbers = []
            metric_values = []
            
            for fold_result in fold_metrics:
                fold_numbers.append(fold_result['fold'])
                metric_values.append(fold_result['metrics'].get(metric, 0))
            
            bars = ax.bar(fold_numbers, metric_values, color=plt.cm.Set3(fold_numbers))
            
            # Add mean line
            mean_value = np.mean(metric_values)
            ax.axhline(y=mean_value, color='red', linestyle='--', 
                      label=f'Mean: {mean_value:.3f}')
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_xlabel('Fold')
            ax.set_ylabel('Score')
            ax.set_title(f'{metric.replace("_", " ").title()} by Fold')
            ax.set_xticks(fold_numbers)
            ax.set_ylim([0, 1.1])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Cross-Validation Results', fontsize=14)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'cross_validation_results.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Cross-validation plot saved to: {save_path}")
    
    def _create_final_report(self, splits, vectorized_data, history, 
                           evaluation_results, cv_results, total_time):
        """Create final training report"""
        
        report = {
            'project_info': {
                'name': 'Email Spam Detection ANN',
                'course': 'Artificial Neural Networks (BSE-635)',
                'instructor': 'Dr. Aamir Arsalan',
                'timestamp': datetime.now().isoformat()
            },
            'data_statistics': {
                'total_samples': len(self.X_train) + len(self.X_val) + len(self.X_test),
                'train_samples': len(self.X_train),
                'val_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'spam_percentage_train': float(self.y_train.mean()),
                'spam_percentage_val': float(self.y_val.mean()),
                'spam_percentage_test': float(self.y_test.mean())
            },
            'model_configuration': {
                'input_dim': self.ann_model.input_dim,
                'hidden_layers': self.ann_model.hidden_layers,
                'activation': self.ann_model.activation,
                'output_activation': self.ann_model.output_activation,
                'dropout_rate': self.ann_model.dropout_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.ann_model.learning_rate,
                'optimizer': self.ann_model.optimizer_name
            },
            'training_info': {
                'total_time_seconds': total_time,
                'final_epoch': len(history.history['loss']) if history else 0,
                'final_train_loss': history.history['loss'][-1] if history else 0,
                'final_val_loss': history.history['val_loss'][-1] if history and 'val_loss' in history.history else 0
            },
            'evaluation_results': evaluation_results,
            'cross_validation_results': cv_results,
            'summary': {
                'best_test_accuracy': evaluation_results['test']['accuracy'] if evaluation_results else 0,
                'best_test_f1': evaluation_results['test']['f1_score'] if evaluation_results else 0,
                'cv_mean_accuracy': cv_results['mean_metrics']['accuracy'] if cv_results else 0,
                'overall_performance': 'EXCELLENT' if evaluation_results and evaluation_results['test']['accuracy'] > 0.9 else
                                      'GOOD' if evaluation_results and evaluation_results['test']['accuracy'] > 0.8 else
                                      'SATISFACTORY' if evaluation_results and evaluation_results['test']['accuracy'] > 0.7 else
                                      'NEEDS IMPROVEMENT'
            }
        }
        
        # Save report
        report_path = self.reports_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create text summary
        summary_path = self.reports_dir / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EMAIL SPAM DETECTION ANN - TRAINING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("PROJECT INFORMATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Course: {report['project_info']['course']}\n")
            f.write(f"Instructor: {report['project_info']['instructor']}\n")
            f.write(f"Timestamp: {report['project_info']['timestamp']}\n\n")
            
            f.write("DATA STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Samples: {report['data_statistics']['total_samples']}\n")
            f.write(f"Train Samples: {report['data_statistics']['train_samples']}\n")
            f.write(f"Validation Samples: {report['data_statistics']['val_samples']}\n")
            f.write(f"Test Samples: {report['data_statistics']['test_samples']}\n")
            f.write(f"Spam Percentage (Train): {report['data_statistics']['spam_percentage_train']:.2%}\n")
            f.write(f"Spam Percentage (Test): {report['data_statistics']['spam_percentage_test']:.2%}\n\n")
            
            f.write("MODEL CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Input Dimension: {report['model_configuration']['input_dim']}\n")
            f.write(f"Hidden Layers: {report['model_configuration']['hidden_layers']}\n")
            f.write(f"Activation: {report['model_configuration']['activation']}\n")
            f.write(f"Dropout Rate: {report['model_configuration']['dropout_rate']}\n")
            f.write(f"Batch Size: {report['model_configuration']['batch_size']}\n")
            f.write(f"Epochs: {report['model_configuration']['epochs']}\n")
            f.write(f"Learning Rate: {report['model_configuration']['learning_rate']}\n")
            f.write(f"Optimizer: {report['model_configuration']['optimizer']}\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-"*40 + "\n")
            f.write(f"Test Accuracy: {report['summary']['best_test_accuracy']:.4f}\n")
            f.write(f"Test F1-Score: {report['summary']['best_test_f1']:.4f}\n")
            if cv_results:
                f.write(f"CV Mean Accuracy: {report['summary']['cv_mean_accuracy']:.4f}\n")
            f.write(f"Training Time: {report['training_info']['total_time_seconds']:.2f} seconds\n")
            f.write(f"Overall Performance: {report['summary']['overall_performance']}\n\n")
            
            f.write("EVALUATION METRICS (Test Set):\n")
            f.write("-"*40 + "\n")
            if evaluation_results and 'test' in evaluation_results:
                test_metrics = evaluation_results['test']
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                    if metric in test_metrics:
                        f.write(f"{metric.replace('_', ' ').title():<15}: {test_metrics[metric]:.4f}\n")
        
        self.logger.info(f"Final report saved to: {report_path}")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return report

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Run full pipeline
    report = trainer.run_full_pipeline()
    
    print("\nTraining completed successfully!")
    print(f"Test Accuracy: {report['summary']['best_test_accuracy']:.4f}")
    print(f"Overall Performance: {report['summary']['overall_performance']}")