"""
Inference modules for Email Spam Detection ANN
"""

__version__ = "1.0.0"
__author__ = "ANN Project Team"

from .predictor import SpamPredictor
from .evaluate import ModelEvaluator

__all__ = ['SpamPredictor', 'ModelEvaluator']