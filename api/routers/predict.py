"""
Prediction endpoints for Email Spam Detection API
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
from typing import List, Optional
import tempfile
import json
from datetime import datetime
from pathlib import Path

from api.schemas import (
    PredictionRequest, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse,
    FilePredictionRequest, FilePredictionResponse,
    EvaluationRequest, EvaluationResponse
)
from inference.predictor import SpamPredictor

router = APIRouter(prefix="/predict")

# Global predictor instance (injected from main)
predictor: SpamPredictor = None

def get_predictor():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return predictor

@router.post("", response_model=PredictionResponse)
async def predict_spam(request: PredictionRequest):
    """
    Predict spam for a single text
    """
    try:
        pred = get_predictor()
        
        # Make prediction
        result = pred.predict_single(request.text, request.threshold)
        
        return PredictionResponse(
            text=request.text,
            prediction=result['prediction'],
            probability=result['probability'],
            is_spam=result['class'] == 'SPAM',
            confidence=result['confidence'],
            cleaned_text=result.get('cleaned_text_preview', ''),
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict spam for multiple texts
    """
    try:
        pred = get_predictor()
        
        # Make predictions
        results = pred.predict_batch(request.texts, request.threshold)
        
        # Format response
        predictions = []
        for result in results:
            predictions.append({
                'text': result['text_preview'],
                'prediction': result['prediction'],
                'probability': result['probability'],
                'is_spam': result['class'] == 'SPAM',
                'confidence': result['confidence'],
                'cleaned_text': result.get('cleaned_text_preview', '')
            })
        
        # Calculate statistics
        spam_count = sum(1 for r in results if r['prediction'] == 1)
        ham_count = len(results) - spam_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_texts=len(results),
            spam_count=spam_count,
            ham_count=ham_count,
            spam_percentage=(spam_count / len(results)) * 100 if results else 0,
            avg_confidence=avg_confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.post("/file")
async def predict_file(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_predictions: bool = Form(False)
):
    """
    Predict spam for texts in a file
    """
    try:
        pred = get_predictor()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Make predictions
            result = pred.predict_file(tmp_path, threshold)
            
            response_data = {
                'file_name': result['file_name'],
                'total_texts': result['total_texts'],
                'spam_count': result['spam_count'],
                'ham_count': result['ham_count'],
                'spam_percentage': result['spam_percentage'],
                'avg_confidence': result['avg_confidence'],
                'timestamp': result['timestamp']
            }
            
            if return_predictions:
                response_data['predictions'] = result['predictions']
            
            return JSONResponse(content=response_data)
            
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_predictions(request: EvaluationRequest):
    """
    Evaluate predictions with ground truth labels
    """
    try:
        pred = get_predictor()
        
        if len(request.texts) != len(request.labels):
            raise HTTPException(
                status_code=400, 
                detail=f"Number of texts ({len(request.texts)}) doesn't match number of labels ({len(request.labels)})"
            )
        
        # Evaluate predictions
        result = pred.evaluate_with_labels(request.texts, request.labels, request.threshold)
        
        return EvaluationResponse(
            total_samples=result['total_samples'],
            accuracy=result['accuracy'],
            precision=result['precision'],
            recall=result['recall'],
            f1_score=result['f1_score'],
            roc_auc=result.get('roc_auc', 0.0),
            correct_predictions=result['correct_predictions'],
            incorrect_predictions=result['incorrect_predictions'],
            false_positives=result['false_positives'],
            false_negatives=result['false_negatives'],
            confusion_matrix=result['confusion_matrix'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.post("/analyze/errors")
async def analyze_errors(request: EvaluationRequest):
    """
    Analyze prediction errors
    """
    try:
        pred = get_predictor()
        
        if len(request.texts) != len(request.labels):
            raise HTTPException(
                status_code=400, 
                detail=f"Number of texts ({len(request.texts)}) doesn't match number of labels ({len(request.labels)})"
            )
        
        # Analyze errors
        result = pred.analyze_errors(request.texts, request.labels, request.threshold)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analysis failed: {str(e)}")

@router.get("/stats")
async def get_prediction_stats():
    """
    Get prediction statistics
    """
    try:
        pred = get_predictor()
        stats = pred.get_prediction_stats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/threshold/sensitivity")
async def threshold_sensitivity(
    thresholds: Optional[List[float]] = None,
    test_size: int = 100
):
    """
    Analyze threshold sensitivity
    """
    try:
        # This would require test data - for simplicity, using a mock response
        # In a real implementation, you would load test data
        
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Mock response for demonstration
        results = []
        for i, threshold in enumerate(thresholds):
            accuracy = 0.8 + (i * 0.02) - (abs(threshold - 0.5) * 0.1)
            precision = 0.75 + (i * 0.015) - (abs(threshold - 0.5) * 0.08)
            recall = 0.85 + (i * 0.025) - (abs(threshold - 0.5) * 0.12)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'accuracy': max(0.5, min(1.0, accuracy)),
                'precision': max(0.5, min(1.0, precision)),
                'recall': max(0.5, min(1.0, recall)),
                'f1_score': max(0.5, min(1.0, f1))
            })
        
        # Find optimal threshold
        optimal_idx = max(range(len(results)), key=lambda i: results[i]['f1_score'])
        optimal = results[optimal_idx]
        
        return JSONResponse(content={
            'analysis': results,
            'optimal_threshold': optimal['threshold'],
            'optimal_metrics': optimal,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Threshold analysis failed: {str(e)}")