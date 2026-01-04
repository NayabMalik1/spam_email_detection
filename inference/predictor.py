"""
Spam prediction module for trained ANN model
"""

import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

from models.ann_model import ANNModel
from models.text_vectorizer import TextVectorizer
from utils.text_cleaner import TextCleaner
from utils.metrics import MetricsCalculator

class SpamPredictor:
    """
    Spam prediction system using trained ANN model
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        if config_path is None:
            config_path = (Path(__file__).parent.parent / 'configs' / 'config.yaml').resolve()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        """
        Initialize SpamPredictor
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.vectorizer = None
        self.ann_model = None
        
        # Paths
        self.models_dir = Path(self.config['paths']['models'])
        self.models_dir = (Path(__file__).parent.parent / self.models_dir).resolve()
        self.reports_dir = Path(self.config['paths']['reports']).resolve()

        
       # Debug
        print("Models directory:", self.models_dir)
        print("Files found:", list(self.models_dir.glob('*')))
        
        # Create directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
       
        
        # Load models flag
        self.models_loaded = False
        
        # Prediction statistics
        self.prediction_stats = {
            'total_predictions': 0,
            'spam_predictions': 0,
            'ham_predictions': 0,
            'avg_confidence': 0.0,
            'prediction_history': []
        }
    
    def load_models(self, model_path=None, vectorizer_path=None):
        """
        Load trained models
        
        Args:
            model_path: Path to ANN model
            vectorizer_path: Path to vectorizer
            
        Returns:
            True if models loaded successfully
        """
        try:
            # Find latest model if path not specified
            if model_path is None:
                model_files = list(self.models_dir.glob('*.h5'))
                if not model_files:
                    raise FileNotFoundError(f"No model files found in {self.models_dir}")
                
                # Get most recent model
                model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            
            if vectorizer_path is None:
                vectorizer_files = list(self.models_dir.glob('vectorizer_*.pkl'))
                if not vectorizer_files:
                    raise FileNotFoundError(f"No vectorizer files found in {self.models_dir}")
                
                # Get most recent vectorizer
                vectorizer_path = max(vectorizer_files, key=lambda x: x.stat().st_mtime)
            
            self.logger.info(f"Loading model from: {model_path}")
            self.logger.info(f"Loading vectorizer from: {vectorizer_path}")
            
            # Load vectorizer
            self.vectorizer = TextVectorizer()
            self.vectorizer.load_vectorizer(vectorizer_path)
            
            # Load ANN model
            self.ann_model = ANNModel()
            self.ann_model.load_model(model_path)
            
            self.models_loaded = True
            
            self.logger.info("Models loaded successfully")
            self.logger.info(f"Vectorizer method: {self.vectorizer.method}")
            self.logger.info(f"Model input dimension: {self.ann_model.input_dim}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}", exc_info=True)
            return False
    
    def preprocess_text(self, text: Union[str, List[str]]) -> List[str]:
        """
        Preprocess text for prediction
        
        Args:
            text: Input text or list of texts
            
        Returns:
            List of cleaned texts
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Clean texts
        cleaned_texts = self.text_cleaner.clean_batch(texts, show_progress=False)
        
        return cleaned_texts
    
    def vectorize_text(self, texts: List[str]) -> np.ndarray:
        """
        Vectorize cleaned texts
        
        Args:
            texts: List of cleaned texts
            
        Returns:
            Vectorized texts as numpy array
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        return self.vectorizer.transform(texts)
    
    def predict_single(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict spam for a single text
        
        Args:
            text: Input text
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction results
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text(text)[0]
            
            # Vectorize text
            text_vector = self.vectorize_text([cleaned_text])
            
            # Make prediction
            result = self.ann_model.predict_single(text_vector, threshold)
            
            # Add additional information
            result['text_preview'] = text[:100] + "..." if len(text) > 100 else text
            result['cleaned_text_preview'] = cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text
            result['timestamp'] = datetime.now().isoformat()
            result['threshold_used'] = threshold
            
            # Update statistics
            self._update_prediction_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'prediction': -1,
                'class': 'ERROR',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Predict spam for a batch of texts
        
        Args:
            texts: List of input texts
            threshold: Classification threshold
            
        Returns:
            List of prediction results
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            # Preprocess texts
            cleaned_texts = self.preprocess_text(texts)
            
            # Vectorize texts
            text_vectors = self.vectorize_text(cleaned_texts)
            
            # Make predictions
            probabilities, predictions = self.ann_model.predict(text_vectors, threshold)
            
            # Create results
            results = []
            for i, (text, cleaned_text, prob, pred) in enumerate(zip(texts, cleaned_texts, 
                                                                   probabilities, predictions)):
                result = {
                    'id': i,
                    'probability': float(prob),
                    'prediction': int(pred),
                    'class': 'SPAM' if pred == 1 else 'HAM',
                    'confidence': float(abs(prob - 0.5) * 2),
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                    'cleaned_text_preview': cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text,
                    'timestamp': datetime.now().isoformat(),
                    'threshold_used': threshold
                }
                
                results.append(result)
                
                # Update statistics
                self._update_prediction_stats(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}", exc_info=True)
            raise
    
    def predict_file(self, file_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict spam for texts in a file
        
        Args:
            file_path: Path to input file
            threshold: Classification threshold
            
        Returns:
            Dictionary with file prediction results
        """
        try:
            # Read file based on extension
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            texts = []
            
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read lines, skip empty lines
                    texts = [line.strip() for line in f if line.strip()]
            
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                # Try to find text column
                possible_columns = ['text', 'email', 'message', 'content', 'body']
                text_column = None
                
                for col in possible_columns:
                    if col in df.columns:
                        text_column = col
                        break
                
                if text_column is None:
                    # Use first column that looks like text
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            text_column = col
                            break
                
                if text_column is None:
                    raise ValueError("No text column found in CSV file")
                
                texts = df[text_column].fillna('').astype(str).tolist()
            
            else:
                # Try to read as text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Split by paragraphs
                    texts = [para.strip() for para in content.split('\n\n') if para.strip()]
            
            if not texts:
                raise ValueError("No text content found in file")
            
            # Make predictions
            results = self.predict_batch(texts, threshold)
            
            # Calculate file-level statistics
            spam_count = sum(1 for r in results if r['prediction'] == 1)
            ham_count = len(results) - spam_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            file_result = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'total_texts': len(texts),
                'spam_count': spam_count,
                'ham_count': ham_count,
                'spam_percentage': (spam_count / len(texts)) * 100 if texts else 0,
                'avg_confidence': avg_confidence,
                'predictions': results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save file results
            self._save_file_results(file_result)
            
            return file_result
            
        except Exception as e:
            self.logger.error(f"Error in file prediction: {str(e)}", exc_info=True)
            raise
    
    def evaluate_with_labels(self, texts: List[str], labels: List[int], 
                           threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate predictions with ground truth labels
        
        Args:
            texts: List of input texts
            labels: Ground truth labels (0 for ham, 1 for spam)
            threshold: Classification threshold
            
        Returns:
            Dictionary with evaluation results
        """
        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) doesn't match number of labels ({len(labels)})")
        
        # Make predictions
        predictions = self.predict_batch(texts, threshold)
        
        # Extract predicted labels and probabilities
        y_pred = [p['prediction'] for p in predictions]
        y_pred_proba = [p['probability'] for p in predictions]
        
        # Calculate metrics
        calculator = MetricsCalculator(model_name="Spam Predictor")
        metrics = calculator.calculate_all_metrics(labels, y_pred, y_pred_proba)
        
        # Create detailed results
        detailed_results = []
        for i, (text, label, pred_result) in enumerate(zip(texts, labels, predictions)):
            detailed_result = {
                'id': i,
                'text_preview': text[:50] + "..." if len(text) > 50 else text,
                'true_label': int(label),
                'true_class': 'SPAM' if label == 1 else 'HAM',
                'predicted_label': pred_result['prediction'],
                'predicted_class': pred_result['class'],
                'probability': pred_result['probability'],
                'confidence': pred_result['confidence'],
                'correct': int(label) == pred_result['prediction'],
                'error_type': None
            }
            
            # Determine error type
            if not detailed_result['correct']:
                if label == 0 and pred_result['prediction'] == 1:
                    detailed_result['error_type'] = 'FALSE_POSITIVE'
                else:
                    detailed_result['error_type'] = 'FALSE_NEGATIVE'
            
            detailed_results.append(detailed_result)
        
        # Create evaluation report
        evaluation_report = {
            'total_samples': len(texts),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics.get('roc_auc', 0.0),
            'confusion_matrix': metrics['confusion_matrix'],
            'correct_predictions': sum(1 for r in detailed_results if r['correct']),
            'incorrect_predictions': len(texts) - sum(1 for r in detailed_results if r['correct']),
            'false_positives': sum(1 for r in detailed_results if r['error_type'] == 'FALSE_POSITIVE'),
            'false_negatives': sum(1 for r in detailed_results if r['error_type'] == 'FALSE_NEGATIVE'),
            'detailed_results': detailed_results,
            'metrics': metrics,
            'threshold_used': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save evaluation report
        self._save_evaluation_report(evaluation_report)
        
        # Create evaluation plots
        self._create_evaluation_plots(labels, y_pred, y_pred_proba, metrics)
        
        return evaluation_report
    
    def analyze_errors(self, texts: List[str], labels: List[int], 
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze prediction errors
        
        Args:
            texts: List of input texts
            labels: Ground truth labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with error analysis
        """
        # Get evaluation results
        eval_results = self.evaluate_with_labels(texts, labels, threshold)
        detailed_results = eval_results['detailed_results']
        
        # Separate errors
        false_positives = [r for r in detailed_results if r['error_type'] == 'FALSE_POSITIVE']
        false_negatives = [r for r in detailed_results if r['error_type'] == 'FALSE_NEGATIVE']
        
        # Analyze false positives (ham classified as spam)
        fp_analysis = {
            'count': len(false_positives),
            'avg_confidence': np.mean([r['confidence'] for r in false_positives]) if false_positives else 0,
            'avg_probability': np.mean([r['probability'] for r in false_positives]) if false_positives else 0,
            'examples': []
        }
        
        # Get top false positives (highest confidence errors)
        for fp in sorted(false_positives, key=lambda x: x['confidence'], reverse=True)[:5]:
            fp_analysis['examples'].append({
                'text': fp['text_preview'],
                'probability': fp['probability'],
                'confidence': fp['confidence']
            })
        
        # Analyze false negatives (spam classified as ham)
        fn_analysis = {
            'count': len(false_negatives),
            'avg_confidence': np.mean([r['confidence'] for r in false_negatives]) if false_negatives else 0,
            'avg_probability': np.mean([r['probability'] for r in false_negatives]) if false_negatives else 0,
            'examples': []
        }
        
        # Get top false negatives (lowest probability for spam)
        for fn in sorted(false_negatives, key=lambda x: x['probability'])[:5]:
            fn_analysis['examples'].append({
                'text': fn['text_preview'],
                'probability': fn['probability'],
                'confidence': fn['confidence']
            })
        
        # Error analysis report
        error_report = {
            'total_errors': eval_results['incorrect_predictions'],
            'error_rate': eval_results['incorrect_predictions'] / eval_results['total_samples'],
            'false_positives': fp_analysis,
            'false_negatives': fn_analysis,
            'threshold_used': threshold,
            'suggested_threshold_adjustment': self._suggest_threshold_adjustment(fp_analysis, fn_analysis),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save error analysis
        self._save_error_analysis(error_report)
        
        return error_report
    
    def _suggest_threshold_adjustment(self, fp_analysis: Dict, fn_analysis: Dict) -> Dict:
        """
        Suggest threshold adjustment based on error analysis
        
        Args:
            fp_analysis: False positive analysis
            fn_analysis: False negative analysis
            
        Returns:
            Dictionary with threshold suggestions
        """
        suggestions = {
            'current_threshold': 0.5,
            'suggested_threshold': 0.5,
            'adjustment_direction': 'none',
            'reason': 'Balanced error distribution',
            'confidence': 0.0
        }
        
        fp_count = fp_analysis['count']
        fn_count = fn_analysis['count']
        
        if fp_count > fn_count * 1.5:  # Too many false positives
            suggestions['suggested_threshold'] = 0.6  # Increase threshold
            suggestions['adjustment_direction'] = 'increase'
            suggestions['reason'] = f'Too many false positives ({fp_count} vs {fn_count})'
            suggestions['confidence'] = min(0.9, fp_count / (fp_count + fn_count))
        
        elif fn_count > fp_count * 1.5:  # Too many false negatives
            suggestions['suggested_threshold'] = 0.4  # Decrease threshold
            suggestions['adjustment_direction'] = 'decrease'
            suggestions['reason'] = f'Too many false negatives ({fn_count} vs {fp_count})'
            suggestions['confidence'] = min(0.9, fn_count / (fp_count + fn_count))
        
        return suggestions
    
    def _update_prediction_stats(self, prediction_result: Dict):
        """
        Update prediction statistics
        
        Args:
            prediction_result: Single prediction result
        """
        self.prediction_stats['total_predictions'] += 1
        
        if prediction_result['prediction'] == 1:
            self.prediction_stats['spam_predictions'] += 1
        else:
            self.prediction_stats['ham_predictions'] += 1
        
        # Update average confidence
        old_total = (self.prediction_stats['total_predictions'] - 1) * self.prediction_stats['avg_confidence']
        self.prediction_stats['avg_confidence'] = (old_total + prediction_result['confidence']) / \
                                                  self.prediction_stats['total_predictions']
        
        # Add to history (keep last 1000)
        self.prediction_stats['prediction_history'].append({
            'timestamp': prediction_result['timestamp'],
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'class': prediction_result['class']
        })
        
        if len(self.prediction_stats['prediction_history']) > 1000:
            self.prediction_stats['prediction_history'] = self.prediction_stats['prediction_history'][-1000:]
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """
        Get prediction statistics
        
        Returns:
            Dictionary with prediction statistics
        """
        stats = self.prediction_stats.copy()
        
        # Calculate percentages
        if stats['total_predictions'] > 0:
            stats['spam_percentage'] = (stats['spam_predictions'] / stats['total_predictions']) * 100
            stats['ham_percentage'] = (stats['ham_predictions'] / stats['total_predictions']) * 100
        else:
            stats['spam_percentage'] = 0.0
            stats['ham_percentage'] = 0.0
        
        # Add timestamp
        stats['timestamp'] = datetime.now().isoformat()
        
        return stats
    
    def _save_file_results(self, file_result: Dict):
        """Save file prediction results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = Path(file_result['file_name']).stem
        
        save_path = self.reports_dir / f'file_predictions_{file_name}_{timestamp}.json'
        
        with open(save_path, 'w') as f:
            json.dump(file_result, f, indent=2)
        
        self.logger.info(f"File prediction results saved to: {save_path}")
    
    def _save_evaluation_report(self, evaluation_report: Dict):
        """Save evaluation report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_path = self.reports_dir / f'evaluation_report_{timestamp}.json'
        
        with open(save_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to: {save_path}")
    
    def _save_error_analysis(self, error_analysis: Dict):
        """Save error analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_path = self.reports_dir / f'error_analysis_{timestamp}.json'
        
        with open(save_path, 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        self.logger.info(f"Error analysis saved to: {save_path}")
    
    def _create_evaluation_plots(self, y_true: List[int], y_pred: List[int], 
                               y_pred_proba: List[float], metrics: Dict):
        """Create evaluation plots"""
        import matplotlib.pyplot as plt
        
        calculator = MetricsCalculator(model_name="Spam Predictor")
        
        # Create plots directory
        plots_dir = self.reports_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot confusion matrix
        fig1 = calculator.plot_confusion_matrix(y_true, y_pred, 
                                               title="Evaluation Confusion Matrix",
                                               save=False)
        if fig1:
            fig1.savefig(plots_dir / f'confusion_matrix_{timestamp}.png', 
                        dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # Plot ROC curve
        if y_pred_proba:
            fig2 = calculator.plot_roc_curve(y_true, y_pred_proba,
                                            title="Evaluation ROC Curve",
                                            save=False)
            if fig2:
                fig2.savefig(plots_dir / f'roc_curve_{timestamp}.png',
                           dpi=300, bbox_inches='tight')
                plt.close(fig2)
        
        # Plot metrics comparison
        fig3 = calculator.plot_metrics_comparison(metrics,
                                                 title="Evaluation Metrics",
                                                 save=False)
        if fig3:
            fig3.savefig(plots_dir / f'metrics_comparison_{timestamp}.png',
                        dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        self.logger.info(f"Evaluation plots saved to: {plots_dir}")

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize predictor
    predictor = SpamPredictor()
    
    # Load models
    if predictor.load_models():
        print("Models loaded successfully")
        
        # Test single prediction
        test_text = "Congratulations! You've won a free iPhone. Click here to claim your prize now!"
        result = predictor.predict_single(test_text)
        
        print("\nSingle Prediction Result:")
        print(f"Text: {result['text_preview']}")
        print(f"Prediction: {result['class']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        # Test batch prediction
        test_texts = [
            "Hello, how are you doing today?",
            "WINNER! You have been selected for a $1000 Walmart gift card.",
            "Meeting scheduled for tomorrow at 2 PM.",
            "Buy cheap Viagra online without prescription!"
        ]
        
        results = predictor.predict_batch(test_texts)
        
        print("\nBatch Predictions:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['text_preview']} -> {res['class']} ({res['probability']:.4f})")
        
        # Get statistics
        stats = predictor.get_prediction_stats()
        print(f"\nPrediction Statistics:")
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Spam predictions: {stats['spam_predictions']} ({stats.get('spam_percentage', 0):.1f}%)")
        print(f"Ham predictions: {stats['ham_predictions']} ({stats.get('ham_percentage', 0):.1f}%)")
        print(f"Average confidence: {stats['avg_confidence']:.2%}")
        
    else:
        print("Failed to load models")