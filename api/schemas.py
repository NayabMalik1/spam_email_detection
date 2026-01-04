"""
Pydantic schemas for Email Spam Detection API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")

class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    text: str
    prediction: int
    probability: float
    is_spam: bool
    confidence: float
    cleaned_text: Optional[str] = None
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction"""
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="List of texts to classify")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")

class BatchPredictionItem(BaseModel):
    """Item schema for batch prediction response"""
    text: str
    prediction: int
    probability: float
    is_spam: bool
    confidence: float
    cleaned_text: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction"""
    predictions: List[BatchPredictionItem]
    total_texts: int
    spam_count: int
    ham_count: int
    spam_percentage: float
    avg_confidence: float
    timestamp: str

class FilePredictionRequest(BaseModel):
    """Request schema for file prediction"""
    file_path: str = Field(..., description="Path to file containing texts")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    return_predictions: bool = Field(False, description="Whether to return individual predictions")

class FilePredictionResponse(BaseModel):
    """Response schema for file prediction"""
    file_name: str
    total_texts: int
    spam_count: int
    ham_count: int
    spam_percentage: float
    avg_confidence: float
    predictions: Optional[List[BatchPredictionItem]] = None
    timestamp: str

class EvaluationRequest(BaseModel):
    """Request schema for evaluation"""
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="List of texts to evaluate")
    labels: List[int] = Field(..., description="Ground truth labels (0=ham, 1=spam)")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")

class EvaluationResponse(BaseModel):
    """Response schema for evaluation"""
    total_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    correct_predictions: int
    incorrect_predictions: int
    false_positives: int
    false_negatives: int
    confusion_matrix: List[List[int]]
    timestamp: str

class ThresholdAnalysisRequest(BaseModel):
    """Request schema for threshold analysis"""
    thresholds: Optional[List[float]] = Field(None, description="List of thresholds to analyze")
    test_size: int = Field(100, ge=10, le=10000, description="Number of samples to use for analysis")

class ThresholdAnalysisResponse(BaseModel):
    """Response schema for threshold analysis"""
    analysis: List[Dict[str, Any]]
    optimal_threshold: float
    optimal_metrics: Dict[str, Any]
    timestamp: str

class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    timestamp: str
    version: str
    details: Optional[Dict[str, Any]] = None

class ErrorAnalysisResponse(BaseModel):
    """Response schema for error analysis"""
    total_errors: int
    error_rate: float
    false_positives: Dict[str, Any]
    false_negatives: Dict[str, Any]
    threshold_used: float
    suggested_threshold_adjustment: Dict[str, Any]
    timestamp: str

class DatasetInfoResponse(BaseModel):
    """Response schema for dataset information"""
    total_samples: int
    splits: List[str]
    statistics: Dict[str, Any]
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    model_name: str
    architecture: Dict[str, Any]
    training_params: Dict[str, Any]
    timestamp: str
    file_name: str

class ReportItem(BaseModel):
    """Schema for report item"""
    name: str
    path: str
    size: int
    modified: str
    type: Optional[str] = None

class ReportsListResponse(BaseModel):
    """Response schema for reports list"""
    report_types: Dict[str, List[ReportItem]]
    text_reports: List[ReportItem]
    total_reports: int
    timestamp: str

class DashboardDataResponse(BaseModel):
    """Response schema for dashboard data"""
    performance_metrics: Dict[str, float]
    prediction_stats: Dict[str, Any]
    model_info: Dict[str, Any]
    dataset_info: Dict[str, Any]
    recent_activity: Dict[str, Any]
    timestamp: str