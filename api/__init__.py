"""
API module for Email Spam Detection ANN
"""

__version__ = "1.0.0"
__author__ = "ANN Project Team"

from .main import app
from .routers.predict import router as predict_router
from .schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest

__all__ = ['app', 'predict_router', 'PredictionRequest', 'PredictionResponse', 'BatchPredictionRequest']