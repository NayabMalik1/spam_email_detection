#!/usr/bin/env python3
"""
Script for training the ANN model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_ann import ModelTrainer
from utils.logger import setup_logger

def main():
    """Main function for model training"""
    logger = setup_logger(__name__)
    
    try:
        logger.info("Starting model training pipeline...")
        
        # Initialize ModelTrainer
        trainer = ModelTrainer()
        
        # Run full training pipeline
        report = trainer.run_full_pipeline()
        
        logger.info("Model training completed successfully!")
        logger.info(f"Test Accuracy: {report['summary']['best_test_accuracy']:.4f}")
        logger.info(f"Overall Performance: {report['summary']['overall_performance']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)