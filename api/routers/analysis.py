"""
Analysis endpoints for Email Spam Detection API
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
from datetime import datetime
from pathlib import Path
import json

router = APIRouter(prefix="/analyze")

@router.get("/wordcloud")
async def get_wordcloud(type: str = "all"):
    """
    Get word cloud images
    """
    try:
        # Path to word cloud images
        plots_dir = Path("outputs/plots")
        
        if type == "spam":
            file_path = plots_dir / "wordcloud_spam.png"
        elif type == "ham":
            file_path = plots_dir / "wordcloud_ham.png"
        else:
            file_path = plots_dir / "wordcloud_all.png"
        
        if file_path.exists():
            return FileResponse(
                file_path,
                media_type="image/png",
                filename=f"wordcloud_{type}.png"
            )
        else:
            # Return a placeholder or generate on the fly
            raise HTTPException(
                status_code=404, 
                detail=f"Word cloud not found. Generate it first by training the model."
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get word cloud: {str(e)}")

@router.get("/features/top")
async def get_top_features(n: int = 20):
    """
    Get top N features by importance
    """
    try:
        # This would require access to the vectorizer
        # For now, return mock data
        features = []
        for i in range(n):
            features.append({
                'feature': f"feature_{i+1}",
                'importance': 1.0 - (i * 0.05),
                'frequency': 1000 - (i * 50)
            })
        
        return JSONResponse(content={
            'top_features': features,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get top features: {str(e)}")

@router.get("/performance")
async def get_performance_metrics():
    """
    Get model performance metrics
    """
    try:
        # Path to evaluation results
        reports_dir = Path("outputs/reports")
        eval_files = list(reports_dir.glob("evaluation_results_*.json"))
        
        if not eval_files:
            raise HTTPException(
                status_code=404, 
                detail="No evaluation results found. Train and evaluate the model first."
            )
        
        # Get latest evaluation results
        latest_file = max(eval_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            eval_results = json.load(f)
        
        # Extract key metrics
        metrics = eval_results.get('metrics', {})
        detailed = eval_results.get('detailed_analysis', {})
        
        response = {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'roc_auc': metrics.get('roc_auc', 0),
            'confusion_matrix': metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
            'error_analysis': detailed.get('error_analysis', {}),
            'confidence_analysis': detailed.get('confidence_analysis', {}),
            'timestamp': eval_results.get('timestamp'),
            'evaluation_date': Path(latest_file).stem.split('_')[-1]
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.get("/plots/{plot_type}")
async def get_plot(plot_type: str):
    """
    Get various analysis plots
    """
    try:
        plots_dir = Path("outputs/plots")
        
        # Map plot types to file names
        plot_files = {
            'confusion_matrix': 'confusion_matrix.png',
            'roc_curve': 'roc_curve.png',
            'precision_recall': 'precision_recall_curve.png',
            'metrics_comparison': 'metrics_comparison.png',
            'training_history': 'training_history.png',
            'threshold_sensitivity': 'threshold_sensitivity.png',
            'model_comparison': 'model_comparison.png',
            'feature_distribution': 'feature_distribution_tfidf.png',
            'architecture': 'model_architecture.png'
        }
        
        if plot_type not in plot_files:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid plot type. Available: {list(plot_files.keys())}"
            )
        
        file_path = plots_dir / plot_files[plot_type]
        
        if file_path.exists():
            return FileResponse(
                file_path,
                media_type="image/png",
                filename=f"{plot_type}.png"
            )
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Plot not found. Generate it first by training/evaluating the model."
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get plot: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """
    Get model information and architecture
    """
    try:
        # Path to model configuration
        models_dir = Path("models/saved_models")
        config_files = list(models_dir.glob("*.json"))
        
        if not config_files:
            raise HTTPException(
                status_code=404, 
                detail="No model configuration found. Train the model first."
            )
        
        # Get latest configuration
        latest_config = max(config_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_config, 'r') as f:
            model_config = json.load(f)
        
        return JSONResponse(content={
            'model_name': model_config.get('model_name', 'Unknown'),
            'architecture': {
                'input_dim': model_config.get('input_dim', 0),
                'hidden_layers': model_config.get('hidden_layers', []),
                'activation': model_config.get('activation', 'relu'),
                'dropout_rate': model_config.get('dropout_rate', 0.0),
                'l2_regularization': model_config.get('l2_regularization', 0.0)
            },
            'training_params': model_config.get('training_params', {}),
            'timestamp': model_config.get('timestamp'),
            'file_name': latest_config.name
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")