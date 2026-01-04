"""
Model evaluation module for trained ANN model
"""

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from models.ann_model import ANNModel
from models.text_vectorizer import TextVectorizer
from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator

class ModelEvaluator:
    """
    Comprehensive model evaluation system
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize ModelEvaluator
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_loader = DataLoader(config_path)
        self.ann_model = ANNModel(config_path)
        self.vectorizer = TextVectorizer(config_path)
        
        # Paths
        self.models_dir = Path(self.config['paths']['models'])
        self.reports_dir = Path(self.config['paths']['reports'])
        self.plots_dir = Path(self.config['paths']['plots'])
        
        # Create directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation data
        self.X_test = None
        self.y_test = None
        self.test_texts = None
        
        # Results storage
        self.evaluation_results = {}
        self.comparison_results = {}
        
    def load_test_data(self):
        """
        Load test data for evaluation
        
        Returns:
            Tuple of (X_test, y_test, test_texts)
        """
        self.logger.info("Loading test data...")
        
        try:
            # Load splits
            splits = self.data_loader.load_splits()
            
            if 'test' not in splits or splits['test'] is None:
                self.logger.error("Test data not found. Please preprocess data first.")
                raise ValueError("Test data not found")
            
            test_df = splits['test']
            
            # Clean text
            from utils.text_cleaner import TextCleaner
            text_cleaner = TextCleaner()
            
            self.logger.info("Cleaning test texts...")
            test_texts = text_cleaner.clean_batch(
                test_df[self.config['data']['text_column']].tolist(),
                show_progress=True
            )
            
            # Load vectorizer
            vectorizer_files = list(self.models_dir.glob('vectorizer_*.pkl'))
            if not vectorizer_files:
                raise FileNotFoundError(f"No vectorizer files found in {self.models_dir}")
            
            vectorizer_path = max(vectorizer_files, key=lambda x: x.stat().st_mtime)
            self.vectorizer.load_vectorizer(vectorizer_path)
            
            # Vectorize texts
            X_test = self.vectorizer.transform(test_texts)
            y_test = test_df[self.config['data']['label_column']].values
            
            self.X_test = X_test
            self.y_test = y_test
            self.test_texts = test_texts
            
            self.logger.info(f"Test data loaded: {X_test.shape}")
            self.logger.info(f"Class distribution: {pd.Series(y_test).value_counts().to_dict()}")
            
            return X_test, y_test, test_texts
            
        except Exception as e:
            self.logger.error(f"Error loading test data: {str(e)}", exc_info=True)
            raise
    
    def load_model(self, model_path=None):
        """
        Load trained model
        
        Args:
            model_path: Path to model file (optional)
            
        Returns:
            Loaded model
        """
        try:
            # Find latest model if path not specified
            if model_path is None:
                model_files = list(self.models_dir.glob('*.h5'))
                if not model_files:
                    raise FileNotFoundError(f"No model files found in {self.models_dir}")
                
                # Get most recent model
                model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            
            self.logger.info(f"Loading model from: {model_path}")
            
            # Load model
            self.ann_model.load_model(model_path)
            
            self.logger.info("Model loaded successfully")
            
            return self.ann_model.model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
    
    def evaluate_model(self, threshold=0.5):
        """
        Evaluate model on test data
        
        Args:
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation results
        """
        if self.X_test is None or self.y_test is None:
            self.load_test_data()
        
        if self.ann_model.model is None:
            self.load_model()
        
        self.logger.info("Evaluating model on test data...")
        
        try:
            # Get predictions
            y_pred_proba = self.ann_model.model.predict(self.X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate metrics
            calculator = MetricsCalculator(model_name=self.config['model']['name'])
            metrics = calculator.calculate_all_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Create detailed analysis
            detailed_analysis = self._create_detailed_analysis(self.y_test, y_pred, y_pred_proba, 
                                                             self.test_texts)
            
            # Store results
            self.evaluation_results = {
                'model_name': self.config['model']['name'],
                'threshold': threshold,
                'metrics': metrics,
                'detailed_analysis': detailed_analysis,
                'timestamp': datetime.now().isoformat(),
                'test_set_size': len(self.X_test),
                'class_distribution': {
                    'ham_count': int(sum(1 for y in self.y_test if y == 0)),
                    'spam_count': int(sum(1 for y in self.y_test if y == 1)),
                    'ham_percentage': float(sum(1 for y in self.y_test if y == 0) / len(self.y_test) * 100),
                    'spam_percentage': float(sum(1 for y in self.y_test if y == 1) / len(self.y_test) * 100)
                }
            }
            
            # Save evaluation results
            self._save_evaluation_results()
            
            # Create evaluation plots
            self._create_evaluation_plots(self.y_test, y_pred, y_pred_proba, metrics)
            
            # Generate report
            self._generate_evaluation_report()
            
            self.logger.info("Model evaluation completed successfully")
            self.logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
            
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
            raise
    
    def evaluate_threshold_sensitivity(self, thresholds=None):
        """
        Evaluate model performance across different thresholds
        
        Args:
            thresholds: List of thresholds to evaluate (optional)
            
        Returns:
            DataFrame with threshold sensitivity analysis
        """
        if self.X_test is None or self.y_test is None:
            self.load_test_data()
        
        if self.ann_model.model is None:
            self.load_model()
        
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        self.logger.info(f"Evaluating threshold sensitivity for {len(thresholds)} thresholds...")
        
        # Get probabilities
        y_pred_proba = self.ann_model.model.predict(self.X_test, verbose=0).flatten()
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            # Calculate confusion matrix elements
            tn = np.sum((y_pred == 0) & (self.y_test == 0))
            fp = np.sum((y_pred == 1) & (self.y_test == 0))
            fn = np.sum((y_pred == 0) & (self.y_test == 1))
            tp = np.sum((y_pred == 1) & (self.y_test == 1))
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            })
        
        # Create DataFrame
        df_thresholds = pd.DataFrame(results)
        
        # Find optimal threshold (max F1-score)
        optimal_idx = df_thresholds['f1_score'].idxmax()
        optimal_threshold = df_thresholds.loc[optimal_idx, 'threshold']
        
        self.logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        self.logger.info(f"Optimal F1-score: {df_thresholds.loc[optimal_idx, 'f1_score']:.4f}")
        
        # Save threshold analysis
        self._save_threshold_analysis(df_thresholds, optimal_threshold)
        
        # Plot threshold sensitivity
        self._plot_threshold_sensitivity(df_thresholds, optimal_threshold)
        
        return df_thresholds, optimal_threshold
    
    def compare_with_baseline(self, baseline_methods=None):
        """
        Compare ANN model with baseline methods
        
        Args:
            baseline_methods: List of baseline methods to compare (optional)
            
        Returns:
            Dictionary with comparison results
        """
        if self.X_test is None or self.y_test is None:
            self.load_test_data()
        
        self.logger.info("Comparing ANN model with baseline methods...")
        
        if baseline_methods is None:
            baseline_methods = ['naive_bayes', 'logistic_regression', 'random_forest', 'svm']
        
        comparison_results = {}
        
        # ANN model results (already evaluated)
        if self.evaluation_results:
            ann_metrics = self.evaluation_results['metrics']
            comparison_results['ann'] = {
                'accuracy': ann_metrics['accuracy'],
                'precision': ann_metrics['precision'],
                'recall': ann_metrics['recall'],
                'f1_score': ann_metrics['f1_score'],
                'roc_auc': ann_metrics.get('roc_auc', 0.0)
            }
        
        # Baseline methods
        for method in baseline_methods:
            try:
                self.logger.info(f"Evaluating baseline: {method}")
                metrics = self._evaluate_baseline(method)
                comparison_results[method] = metrics
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {method}: {str(e)}")
                comparison_results[method] = None
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_results).T
        
        # Sort by F1-score
        if 'f1_score' in df_comparison.columns:
            df_comparison = df_comparison.sort_values('f1_score', ascending=False)
        
        self.comparison_results = {
            'comparison_df': df_comparison.to_dict(),
            'best_method': df_comparison.index[0] if not df_comparison.empty else None,
            'best_f1_score': df_comparison['f1_score'].iloc[0] if 'f1_score' in df_comparison.columns and not df_comparison.empty else 0,
            'ann_rank': df_comparison.index.get_loc('ann') + 1 if 'ann' in df_comparison.index else None
        }
        
        # Save comparison results
        self._save_comparison_results()
        
        # Plot comparison
        self._plot_model_comparison(df_comparison)
        
        self.logger.info(f"\nModel Comparison Results:")
        self.logger.info(df_comparison.to_string())
        
        return self.comparison_results
    
    def _evaluate_baseline(self, method):
        """
        Evaluate baseline method
        
        Args:
            method: Baseline method name
            
        Returns:
            Dictionary with metrics
        """
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Load or create vectorized training data
        try:
            splits = self.data_loader.load_splits()
            
            # Clean and vectorize training data
            from utils.text_cleaner import TextCleaner
            text_cleaner = TextCleaner()
            
            train_texts = text_cleaner.clean_batch(
                splits['train'][self.config['data']['text_column']].tolist(),
                show_progress=False
            )
            
            val_texts = text_cleaner.clean_batch(
                splits['val'][self.config['data']['text_column']].tolist(),
                show_progress=False
            )
            
            # Combine train and val for baseline training
            all_train_texts = train_texts + val_texts
            
            # Load vectorizer
            if self.vectorizer.vectorizer is None:
                vectorizer_files = list(self.models_dir.glob('vectorizer_*.pkl'))
                if vectorizer_files:
                    vectorizer_path = max(vectorizer_files, key=lambda x: x.stat().st_mtime)
                    self.vectorizer.load_vectorizer(vectorizer_path)
            
            # Vectorize texts
            X_train = self.vectorizer.transform(all_train_texts)
            y_train = np.concatenate([
                splits['train'][self.config['data']['label_column']].values,
                splits['val'][self.config['data']['label_column']].values
            ])
            
            X_test = self.vectorizer.transform(self.test_texts)
            
        except Exception as e:
            self.logger.warning(f"Using test data for baseline {method}: {str(e)}")
            X_train = self.X_test
            y_train = self.y_test
            X_test = self.X_test
        
        # Initialize and train baseline model
        if method == 'naive_bayes':
            model = MultinomialNB()
        elif method == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif method == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif method == 'svm':
            model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown baseline method: {method}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0)
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _create_detailed_analysis(self, y_true, y_pred, y_pred_proba, texts):
        """
        Create detailed analysis of predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            texts: Original texts
            
        Returns:
            Dictionary with detailed analysis
        """
        # Calculate per-class statistics
        ham_indices = np.where(y_true == 0)[0]
        spam_indices = np.where(y_true == 1)[0]
        
        ham_probs = y_pred_proba[ham_indices]
        spam_probs = y_pred_proba[spam_indices]
        
        # Error analysis
        errors = y_pred != y_true
        error_indices = np.where(errors)[0]
        
        false_positives = [i for i in error_indices if y_true[i] == 0 and y_pred[i] == 1]
        false_negatives = [i for i in error_indices if y_true[i] == 1 and y_pred[i] == 0]
        
        # Confidence analysis
        confidences = np.abs(y_pred_proba - 0.5) * 2  # Normalize to [0, 1]
        
        detailed_analysis = {
            'per_class_statistics': {
                'ham': {
                    'count': len(ham_indices),
                    'mean_probability': float(np.mean(ham_probs)) if len(ham_probs) > 0 else 0,
                    'std_probability': float(np.std(ham_probs)) if len(ham_probs) > 0 else 0,
                    'correct_predictions': int(np.sum(y_pred[ham_indices] == 0)),
                    'incorrect_predictions': int(np.sum(y_pred[ham_indices] == 1))
                },
                'spam': {
                    'count': len(spam_indices),
                    'mean_probability': float(np.mean(spam_probs)) if len(spam_probs) > 0 else 0,
                    'std_probability': float(np.std(spam_probs)) if len(spam_probs) > 0 else 0,
                    'correct_predictions': int(np.sum(y_pred[spam_indices] == 1)),
                    'incorrect_predictions': int(np.sum(y_pred[spam_indices] == 0))
                }
            },
            'error_analysis': {
                'total_errors': len(error_indices),
                'false_positives': len(false_positives),
                'false_negatives': len(false_negatives),
                'error_rate': len(error_indices) / len(y_true),
                'fp_rate': len(false_positives) / len(ham_indices) if len(ham_indices) > 0 else 0,
                'fn_rate': len(false_negatives) / len(spam_indices) if len(spam_indices) > 0 else 0
            },
            'confidence_analysis': {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'min_confidence': float(np.min(confidences)),
                'max_confidence': float(np.max(confidences)),
                'confidence_by_class': {
                    'ham': float(np.mean(confidences[ham_indices])) if len(ham_indices) > 0 else 0,
                    'spam': float(np.mean(confidences[spam_indices])) if len(spam_indices) > 0 else 0
                }
            },
            'error_examples': {
                'false_positives': [],
                'false_negatives': []
            }
        }
        
        # Add example errors
        for i in false_positives[:5]:  # Top 5 false positives
            detailed_analysis['error_examples']['false_positives'].append({
                'text_preview': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                'probability': float(y_pred_proba[i]),
                'confidence': float(confidences[i])
            })
        
        for i in false_negatives[:5]:  # Top 5 false negatives
            detailed_analysis['error_examples']['false_negatives'].append({
                'text_preview': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                'probability': float(y_pred_proba[i]),
                'confidence': float(confidences[i])
            })
        
        return detailed_analysis
    
    def _save_evaluation_results(self):
        """Save evaluation results to file"""
        if not self.evaluation_results:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_path = self.reports_dir / f'evaluation_results_{timestamp}.json'
        
        with open(save_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to: {save_path}")
    
    def _save_threshold_analysis(self, df_thresholds, optimal_threshold):
        """Save threshold analysis to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        analysis = {
            'threshold_analysis': df_thresholds.to_dict(orient='records'),
            'optimal_threshold': float(optimal_threshold),
            'optimal_metrics': df_thresholds[df_thresholds['threshold'] == optimal_threshold].to_dict(orient='records')[0],
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = self.reports_dir / f'threshold_analysis_{timestamp}.json'
        
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Threshold analysis saved to: {save_path}")
    
    def _save_comparison_results(self):
        """Save comparison results to file"""
        if not self.comparison_results:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_path = self.reports_dir / f'comparison_results_{timestamp}.json'
        
        with open(save_path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        self.logger.info(f"Comparison results saved to: {save_path}")
    
    def _create_evaluation_plots(self, y_true, y_pred, y_pred_proba, metrics):
        """Create evaluation plots"""
        calculator = MetricsCalculator(model_name=self.config['model']['name'])
        
        # Plot confusion matrix
        calculator.plot_confusion_matrix(y_true, y_pred, 
                                        title="Test Set Confusion Matrix")
        
        # Plot ROC curve
        if y_pred_proba is not None:
            calculator.plot_roc_curve(y_true, y_pred_proba, 
                                     title="Test Set ROC Curve")
        
        # Plot precision-recall curve
        if y_pred_proba is not None:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, color='darkgreen', lw=2,
                   label=f'Precision-Recall (AP = {avg_precision:.3f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Test Set Precision-Recall Curve')
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            
            save_path = self.plots_dir / 'precision_recall_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot metrics comparison
        calculator.plot_metrics_comparison(metrics,
                                          title="Test Set Performance Metrics")
    
    def _plot_threshold_sensitivity(self, df_thresholds, optimal_threshold):
        """Plot threshold sensitivity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flat
        
        # Plot 1: Accuracy, Precision, Recall, F1 vs Threshold
        ax = axes[0]
        ax.plot(df_thresholds['threshold'], df_thresholds['accuracy'], 
               label='Accuracy', linewidth=2)
        ax.plot(df_thresholds['threshold'], df_thresholds['precision'], 
               label='Precision', linewidth=2)
        ax.plot(df_thresholds['threshold'], df_thresholds['recall'], 
               label='Recall', linewidth=2)
        ax.plot(df_thresholds['threshold'], df_thresholds['f1_score'], 
               label='F1-Score', linewidth=3, color='black')
        
        # Mark optimal threshold
        ax.axvline(x=optimal_threshold, color='red', linestyle='--', 
                  label=f'Optimal (F1={df_thresholds[df_thresholds["threshold"] == optimal_threshold]["f1_score"].iloc[0]:.3f})')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error rates vs Threshold
        ax = axes[1]
        ax.plot(df_thresholds['threshold'], df_thresholds['false_positive_rate'], 
               label='False Positive Rate', linewidth=2, color='red')
        ax.plot(df_thresholds['threshold'], df_thresholds['false_negative_rate'], 
               label='False Negative Rate', linewidth=2, color='blue')
        
        ax.axvline(x=optimal_threshold, color='green', linestyle='--', 
                  label=f'Optimal Threshold')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Error Rate')
        ax.set_title('Error Rates vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Confusion matrix elements vs Threshold
        ax = axes[2]
        ax.plot(df_thresholds['threshold'], df_thresholds['true_positives'], 
               label='True Positives', linewidth=2)
        ax.plot(df_thresholds['threshold'], df_thresholds['false_positives'], 
               label='False Positives', linewidth=2)
        ax.plot(df_thresholds['threshold'], df_thresholds['true_negatives'], 
               label='True Negatives', linewidth=2)
        ax.plot(df_thresholds['threshold'], df_thresholds['false_negatives'], 
               label='False Negatives', linewidth=2)
        
        ax.axvline(x=optimal_threshold, color='black', linestyle='--', 
                  label=f'Optimal Threshold')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Count')
        ax.set_title('Confusion Matrix Elements vs Threshold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Trade-off analysis
        ax = axes[3]
        scatter = ax.scatter(df_thresholds['false_positive_rate'], 
                           df_thresholds['false_negative_rate'],
                           c=df_thresholds['threshold'],
                           s=df_thresholds['f1_score'] * 100,
                           cmap='viridis',
                           alpha=0.7,
                           edgecolors='black')
        
        # Mark optimal point
        optimal_fpr = df_thresholds[df_thresholds['threshold'] == optimal_threshold]['false_positive_rate'].iloc[0]
        optimal_fnr = df_thresholds[df_thresholds['threshold'] == optimal_threshold]['false_negative_rate'].iloc[0]
        ax.scatter(optimal_fpr, optimal_fnr, s=200, color='red', marker='*',
                  label=f'Optimal (Threshold={optimal_threshold:.3f})')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('False Negative Rate')
        ax.set_title('Error Trade-off Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Threshold')
        
        plt.suptitle('Threshold Sensitivity Analysis', fontsize=14)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'threshold_sensitivity.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Threshold sensitivity plot saved to: {save_path}")
    
    def _plot_model_comparison(self, df_comparison):
        """Plot model comparison results"""
        if df_comparison.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flat
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in df_comparison.columns:
                ax = axes[i]
                
                # Sort by metric value
                df_sorted = df_comparison.sort_values(metric, ascending=True)
                
                bars = ax.barh(range(len(df_sorted)), df_sorted[metric], 
                              color=plt.cm.Set3(range(len(df_sorted))))
                
                ax.set_yticks(range(len(df_sorted)))
                ax.set_yticklabels(df_sorted.index)
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_xlim([0, 1.0])
                ax.grid(True, alpha=0.3, axis='x')
                
                # Highlight ANN model
                for j, model_name in enumerate(df_sorted.index):
                    if model_name == 'ann':
                        bars[j].set_color('red')
                        bars[j].set_edgecolor('black')
                        bars[j].set_linewidth(2)
        
        plt.suptitle('Model Comparison Results', fontsize=14)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model comparison plot saved to: {save_path}")
    
    def _generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.reports_dir / f'evaluation_report_{timestamp}.txt'
        
        metrics = self.evaluation_results['metrics']
        detailed = self.evaluation_results['detailed_analysis']
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Model Name: {self.evaluation_results['model_name']}\n")
            f.write(f"Evaluation Date: {self.evaluation_results['timestamp']}\n")
            f.write(f"Test Set Size: {self.evaluation_results['test_set_size']}\n")
            f.write(f"Threshold: {self.evaluation_results['threshold']}\n\n")
            
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-"*40 + "\n")
            dist = self.evaluation_results['class_distribution']
            f.write(f"Ham: {dist['ham_count']} ({dist['ham_percentage']:.1f}%)\n")
            f.write(f"Spam: {dist['spam_count']} ({dist['spam_percentage']:.1f}%)\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
            if 'roc_auc' in metrics:
                f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write("-"*40 + "\n")
            cm = metrics['confusion_matrix']
            f.write(f"              Predicted\n")
            f.write(f"              Ham    Spam\n")
            f.write(f"Actual Ham    {cm[0][0]:<6} {cm[0][1]:<6}\n")
            f.write(f"Actual Spam   {cm[1][0]:<6} {cm[1][1]:<6}\n\n")
            
            f.write("ERROR ANALYSIS:\n")
            f.write("-"*40 + "\n")
            errors = detailed['error_analysis']
            f.write(f"Total Errors: {errors['total_errors']}\n")
            f.write(f"Error Rate:   {errors['error_rate']:.4f}\n")
            f.write(f"False Positives: {errors['false_positives']} (Rate: {errors['fp_rate']:.4f})\n")
            f.write(f"False Negatives: {errors['false_negatives']} (Rate: {errors['fn_rate']:.4f})\n\n")
            
            f.write("CONFIDENCE ANALYSIS:\n")
            f.write("-"*40 + "\n")
            conf = detailed['confidence_analysis']
            f.write(f"Mean Confidence: {conf['mean_confidence']:.4f}\n")
            f.write(f"Std Confidence:  {conf['std_confidence']:.4f}\n")
            f.write(f"Min Confidence:  {conf['min_confidence']:.4f}\n")
            f.write(f"Max Confidence:  {conf['max_confidence']:.4f}\n")
            f.write(f"Ham Confidence:  {conf['confidence_by_class']['ham']:.4f}\n")
            f.write(f"Spam Confidence: {conf['confidence_by_class']['spam']:.4f}\n\n")
            
            f.write("PER-CLASS STATISTICS:\n")
            f.write("-"*40 + "\n")
            stats = detailed['per_class_statistics']
            f.write("Ham Class:\n")
            f.write(f"  Count: {stats['ham']['count']}\n")
            f.write(f"  Mean Probability: {stats['ham']['mean_probability']:.4f}\n")
            f.write(f"  Correct: {stats['ham']['correct_predictions']}\n")
            f.write(f"  Incorrect: {stats['ham']['incorrect_predictions']}\n\n")
            
            f.write("Spam Class:\n")
            f.write(f"  Count: {stats['spam']['count']}\n")
            f.write(f"  Mean Probability: {stats['spam']['mean_probability']:.4f}\n")
            f.write(f"  Correct: {stats['spam']['correct_predictions']}\n")
            f.write(f"  Incorrect: {stats['spam']['incorrect_predictions']}\n\n")
            
            f.write("ERROR EXAMPLES:\n")
            f.write("-"*40 + "\n")
            examples = detailed['error_examples']
            
            if examples['false_positives']:
                f.write("False Positives (Ham classified as Spam):\n")
                for i, ex in enumerate(examples['false_positives'][:3], 1):
                    f.write(f"  {i}. Text: {ex['text_preview']}\n")
                    f.write(f"     Probability: {ex['probability']:.4f}, Confidence: {ex['confidence']:.4f}\n")
                f.write("\n")
            
            if examples['false_negatives']:
                f.write("False Negatives (Spam classified as Ham):\n")
                for i, ex in enumerate(examples['false_negatives'][:3], 1):
                    f.write(f"  {i}. Text: {ex['text_preview']}\n")
                    f.write(f"     Probability: {ex['probability']:.4f}, Confidence: {ex['confidence']:.4f}\n")
        
        self.logger.info(f"Evaluation report saved to: {report_path}")

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate model
    results = evaluator.evaluate_model()
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
    
    # Evaluate threshold sensitivity
    df_thresholds, optimal = evaluator.evaluate_threshold_sensitivity()
    
    print(f"\nOptimal Threshold: {optimal:.3f}")
    print(f"Optimal F1-Score: {df_thresholds[df_thresholds['threshold'] == optimal]['f1_score'].iloc[0]:.4f}")
    
    # Compare with baselines
    comparison = evaluator.compare_with_baseline()
    
    print(f"\nBest Method: {comparison['best_method']}")
    print(f"Best F1-Score: {comparison['best_f1_score']:.4f}")
    print(f"ANN Rank: {comparison['ann_rank']}")