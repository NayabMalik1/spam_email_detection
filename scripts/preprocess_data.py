#!/usr/bin/env python3
"""
Script for preprocessing the spam email dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_loader import DataLoader
from utils.logger import setup_logger

def main():
    """Main function for data preprocessing"""
    logger = setup_logger(__name__)
    
    try:
        logger.info("Starting data preprocessing...")
        
        # Initialize DataLoader
        data_loader = DataLoader()
        
        # Load and preprocess data
        df_raw = data_loader.load_raw_data()
        df_processed, stats = data_loader.preprocess_data(df_raw)
        splits, split_stats = data_loader.split_data(df_processed)
        
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Total samples: {len(df_processed)}")
        logger.info(f"Train samples: {len(splits['train'])}")
        logger.info(f"Validation samples: {len(splits['val'])}")
        logger.info(f"Test samples: {len(splits['test'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)