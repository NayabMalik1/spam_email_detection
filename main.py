#!/usr/bin/env python3
"""
Main entry point for Email Spam Detection ANN Project
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger
from api.main import run_api
from scripts.preprocess_data import main as preprocess_data
from scripts.train_model import main as train_model
from scripts.test_model import main as test_model

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'logs',
        'outputs/plots',
        'outputs/reports',
        'temp'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory created: {dir_path}")

def main():
    """Main function to run the application"""
    # Setup logging
    logger = setup_logger(__name__)
    
    print("\n" + "="*60)
    print("      EMAIL SPAM DETECTION - ANN PROJECT")
    print("="*60)
    print("Course: Artificial Neural Networks (BSE-635)")
    print("Instructor: Dr. Aamir Arsalan")
    print("Semester: VII")
    print("="*60 + "\n")
    
    # Create directories
    setup_directories()
    
    print("\nOptions:")
    print("1. Preprocess Data")
    print("2. Train Model")
    print("3. Test Model")
    print("4. Launch Web Interface")
    print("5. Run All")
    print("6. Exit")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\nPreprocessing data...")
            preprocess_data()
            
        elif choice == '2':
            print("\nTraining model...")
            train_model()
            
        elif choice == '3':
            print("\nTesting model...")
            test_model()
            
        elif choice == '4':
            print("\nLaunching Web Interface...")
            print("Open your browser and navigate to: http://localhost:8000")
            print("Press Ctrl+C to stop the server")
            run_api()
            
        elif choice == '5':
            print("\nRunning complete pipeline...")
            preprocess_data()
            train_model()
            test_model()
            print("\nLaunching Web Interface...")
            print("Open your browser and navigate to: http://localhost:8000")
            run_api()
            
        elif choice == '6':
            print("\nExiting...")
            sys.exit(0)
            
        else:
            print("Invalid choice. Please enter a number between 1-6.")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()