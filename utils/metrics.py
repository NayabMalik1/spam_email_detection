"""
Metrics calculation and visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class MetricsCalculator:
    """
    Comprehensive metrics calculator and visualizer
    """
    
    def __init__(self, model_name="ANN Model", save_dir="outputs/reports"):
        """
        Initialize metrics calculator
        
        Args:
            model_name: Name of the model for reporting
            save_dir: Directory to save reports and plots
        """
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for matplotlib plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Additional metrics if probabilities are provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Matthews correlation coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", save=True):
        """
        Plot and save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], 
                   yticklabels=['Ham', 'Spam'],
                   ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{title}\n{self.model_name}')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba, title="ROC Curve", save=True):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{title}\n{self.model_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'roc_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, title="Precision-Recall Curve", save=True):
        """
        Plot precision-recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color='darkgreen', lw=2,
                label=f'Precision-Recall (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{title}\n{self.model_name}')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'precision_recall_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to: {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, metrics_dict, title="Model Performance Metrics", save=True):
        """
        Plot comparison of different metrics
        
        Args:
            metrics_dict: Dictionary with metric names and values
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            matplotlib figure
        """
        # Filter numeric metrics
        numeric_metrics = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and not key.endswith('_matrix'):
                numeric_metrics[key] = value
        
        if not numeric_metrics:
            print("No numeric metrics found for comparison plot")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(numeric_metrics.keys())
        metric_values = list(numeric_metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=plt.cm.Set3(np.arange(len(metric_names))))
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Score')
        ax.set_title(f'{title}\n{self.model_name}')
        ax.set_ylim([0, 1.1])
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'metrics_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to: {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, y_true, y_pred, y_pred_proba=None, save_html=True):
        """
        Create interactive dashboard with Plotly
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            save_html: Whether to save as HTML
            
        Returns:
            Plotly figure
        """
        # Calculate metrics
        metrics = self.calculate_all_metrics(y_true, y_pred, y_pred_proba)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve',
                          'Metrics Comparison', 'Classification Report', 'Model Performance'),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'bar'}, {'type': 'table'}, {'type': 'indicator'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Ham', 'Spam'],
                y=['Ham', 'Spam'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=1
        )
        
        # 2. ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {metrics["roc_auc"]:.3f})',
                    line=dict(color='darkorange', width=3)
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='navy', width=2, dash='dash')
                ),
                row=1, col=2
            )
            fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
            fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        
        # 3. Precision-Recall Curve (if probabilities available)
        if y_pred_proba is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'PR (AP = {metrics["average_precision"]:.3f})',
                    line=dict(color='darkgreen', width=3)
                ),
                row=1, col=3
            )
            fig.update_xaxes(title_text="Recall", row=1, col=3)
            fig.update_yaxes(title_text="Precision", row=1, col=3)
        
        # 4. Metrics Comparison
        bar_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if 'roc_auc' in metrics:
            bar_metrics.append('roc_auc')
        
        bar_values = [metrics[m] for m in bar_metrics]
        fig.add_trace(
            go.Bar(
                x=bar_metrics,
                y=bar_values,
                text=[f'{v:.3f}' for v in bar_values],
                textposition='auto',
                marker_color=px.colors.qualitative.Set3[:len(bar_metrics)]
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Metrics", row=2, col=1)
        fig.update_yaxes(title_text="Score", range=[0, 1.1], row=2, col=1)
        
        # 5. Classification Report Table
        report_df = pd.DataFrame(metrics['classification_report']).transpose().round(3)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Class'] + list(report_df.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[report_df.index] + [report_df[col] for col in report_df.columns],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=2, col=2
        )
        
        # 6. Performance Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics['accuracy'],
                title={'text': "Overall Accuracy"},
                delta={'reference': 0.5},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.6], 'color': "lightgray"},
                        {'range': [0.6, 0.8], 'color': "gray"},
                        {'range': [0.8, 1], 'color': "lightblue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            width=1400,
            title_text=f"Model Performance Dashboard - {self.model_name}",
            showlegend=False,
            template="plotly_white"
        )
        
        if save_html:
            save_path = self.save_dir / 'interactive_dashboard.html'
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to: {save_path}")
        
        return fig
    
    def generate_report(self, metrics, split_name="test"):
        """
        Generate comprehensive report
        
        Args:
            metrics: Dictionary of metrics
            split_name: Name of the data split
            
        Returns:
            Report as dictionary and saves to file
        """
        report = {
            'model_name': self.model_name,
            'split': split_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': metrics,
            'summary': {
                'best_metric': max([v for k, v in metrics.items() 
                                  if isinstance(v, (int, float)) and not k.endswith('_matrix')]),
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0)
            }
        }
        
        # Save report to JSON
        report_path = self.save_dir / f'{split_name}_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save summary to text file
        summary_path = self.save_dir / f'{split_name}_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"MODEL PERFORMANCE REPORT - {split_name.upper()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"{'-'*60}\n\n")
            
            f.write("KEY METRICS:\n")
            f.write(f"{'-'*40}\n")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'balanced_accuracy']:
                if metric in metrics:
                    f.write(f"{metric.replace('_', ' ').title():<20}: {metrics[metric]:.4f}\n")
            
            f.write(f"\nCONFUSION MATRIX:\n")
            f.write(f"{'-'*40}\n")
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            f.write(f"              Predicted\n")
            f.write(f"              Ham    Spam\n")
            f.write(f"Actual Ham    {cm[0][0]:<6} {cm[0][1]:<6}\n")
            f.write(f"Actual Spam   {cm[1][0]:<6} {cm[1][1]:<6}\n")
            
            f.write(f"\nCLASSIFICATION REPORT:\n")
            f.write(f"{'-'*40}\n")
            if 'classification_report' in metrics:
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                f.write(report_df.to_string())
        
        print(f"Report saved to: {report_path}")
        print(f"Summary saved to: {summary_path}")
        
        return report

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.rand(n_samples)
    
    # Initialize calculator
    calculator = MetricsCalculator(model_name="Test ANN Model")
    
    # Calculate metrics
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    # Create plots
    calculator.plot_confusion_matrix(y_true, y_pred)
    calculator.plot_roc_curve(y_true, y_pred_proba)
    calculator.plot_precision_recall_curve(y_true, y_pred_proba)
    calculator.plot_metrics_comparison(metrics)
    
    # Generate report
    report = calculator.generate_report(metrics, "test")