#!/usr/bin/env python3
"""
Setup script for Email Spam Detection ANN Project
"""

import subprocess
import sys
import nltk

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-eng', quiet=True)
    print("NLTK data downloaded successfully.")

def create_directories():
    """Create necessary directories"""
    import os
    directories = [
        'data/raw',
        'data/processed',
        'models/saved_models',
        'logs',
        'outputs/plots',
        'outputs/reports',
        'temp',
        'api/static',
        'api/templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_environment():
    """Setup environment variables"""
    print("\nSetting up environment...")
    print("✓ All requirements installed")
    print("✓ NLTK data downloaded")
    print("✓ Directories created")
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Place your dataset in data/raw/spam_email.csv")
    print("2. Run: python scripts/preprocess_data.py")
    print("3. Run: python scripts/train_model.py")
    print("4. Run: python main.py to start the web interface")
    print("\nOr run everything at once: python main.py and choose option 5")

if __name__ == "__main__":
    print("="*60)
    print("Email Spam Detection ANN Project - Setup")
    print("="*60)
    
    try:
        install_requirements()
        download_nltk_data()
        create_directories()
        setup_environment()
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        sys.exit(1)