"""
Data loading and splitting utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path
import json

class DataLoader:
    """
    Load and split data for spam detection
    """
    
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize DataLoader with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Data paths
        # Line 33 par:
        self.raw_data_path = "data/raw/spam_email.csv"  # Direct assign karo
        self.raw_data_path = self.config['data']['dataset_path']
        self.processed_dir = Path(self.config['paths']['processed_data'])
        
        # Data parameters
        self.test_size = self.config['data']['test_size']
        self.val_size = self.config['data']['val_size']
        self.random_state = self.config['data']['random_state']
        
        self.text_column = self.config['data']['text_column']
        self.label_column = self.config['data']['label_column']
        
    def load_raw_data(self):
        """
        Load raw data from CSV file
        
        Returns:
            DataFrame with raw data
        """
        self.logger.info(f"Loading raw data from: {self.raw_data_path}")
        
        try:
            # Check if file exists
            if not Path(self.raw_data_path).exists():
                self.logger.error(f"File not found: {self.raw_data_path}")
                raise FileNotFoundError(f"Dataset file not found: {self.raw_data_path}")
            
            # Load data
            df = pd.read_csv(self.raw_data_path)
            
            # Basic validation
            if self.text_column not in df.columns:
                raise ValueError(f"Text column '{self.text_column}' not found in dataset")
            
            if self.label_column not in df.columns:
                # Try to infer label column
                possible_labels = ['label', 'spam', 'is_spam', 'target', 'class']
                for col in possible_labels:
                    if col in df.columns:
                        self.label_column = col
                        break
                
                if self.label_column not in df.columns:
                    raise ValueError(f"Label column '{self.label_column}' not found in dataset")
            
            self.logger.info(f"Loaded {len(df)} records")
            self.logger.info(f"Columns: {list(df.columns)}")
            self.logger.info(f"Label distribution:\n{df[self.label_column].value_counts()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """
        Preprocess data for modeling
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Preprocessing data...")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # 1. Handle missing values
        missing_counts = df_processed.isnull().sum()
        if missing_counts.any():
            self.logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            df_processed = df_processed.dropna(subset=[self.text_column])
        
        # 2. Convert labels to binary (if not already)
        if df_processed[self.label_column].dtype == 'object':
            # Map string labels to binary
            unique_labels = df_processed[self.label_column].unique()
            self.logger.info(f"Unique labels: {unique_labels}")
            
            # Try to infer which label means spam
            label_mapping = {}
            for label in unique_labels:
                label_lower = str(label).lower()
                if any(spam_word in label_lower for spam_word in ['spam', '1', 'yes', 'true']):
                    label_mapping[label] = 1
                elif any(ham_word in label_lower for ham_word in ['ham', '0', 'no', 'false', 'not']):
                    label_mapping[label] = 0
                else:
                    # Default: assume spam is 1
                    label_mapping[label] = int('spam' in label_lower)
            
            df_processed[self.label_column] = df_processed[self.label_column].map(label_mapping)
            self.logger.info(f"Label mapping: {label_mapping}")
        
        # Ensure labels are integers
        df_processed[self.label_column] = df_processed[self.label_column].astype(int)
        
        # 3. Add text length feature
        df_processed['text_length'] = df_processed[self.text_column].apply(len)
        df_processed['word_count'] = df_processed[self.text_column].apply(lambda x: len(str(x).split()))
        
        # 4. Log statistics
        self.logger.info(f"After preprocessing: {len(df_processed)} records")
        self.logger.info(f"Spam distribution: {df_processed[self.label_column].value_counts().to_dict()}")
        
        # Calculate statistics
        stats = {
            'total_samples': len(df_processed),
            'spam_count': int(df_processed[self.label_column].sum()),
            'ham_count': int(len(df_processed) - df_processed[self.label_column].sum()),
            'spam_percentage': round(df_processed[self.label_column].mean() * 100, 2),
            'avg_text_length': round(df_processed['text_length'].mean(), 2),
            'avg_word_count': round(df_processed['word_count'].mean(), 2)
        }
        
        self.logger.info(f"Dataset statistics: {stats}")
        
        return df_processed, stats
    
    def split_data(self, df):
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary with splits and their statistics
        """
        self.logger.info("Splitting data into train/val/test sets...")
        
        # Calculate split sizes
        total_size = len(df)
        test_size = int(total_size * self.test_size)
        val_size = int(total_size * self.val_size)
        train_size = total_size - test_size - val_size
        
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df[self.label_column]
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size/(len(train_val_df)),
            random_state=self.random_state,
            stratify=train_val_df[self.label_column]
        )
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Calculate statistics for each split
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        split_stats = {}
        for split_name, split_df in splits.items():
            stats = {
                'samples': len(split_df),
                'spam_count': int(split_df[self.label_column].sum()),
                'ham_count': int(len(split_df) - split_df[self.label_column].sum()),
                'spam_percentage': round(split_df[self.label_column].mean() * 100, 2),
                'avg_text_length': round(split_df['text_length'].mean(), 2),
                'avg_word_count': round(split_df['word_count'].mean(), 2)
            }
            split_stats[split_name] = stats
            
            self.logger.info(f"{split_name.upper()} set: {stats}")
        
        # Save splits to files
        self.save_splits(splits, split_stats)
        
        return splits, split_stats
    
    def save_splits(self, splits, stats):
        """
        Save splits to CSV files
        
        Args:
            splits: Dictionary of DataFrames
            stats: Statistics dictionary
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        for split_name, split_df in splits.items():
            file_path = self.processed_dir / f'{split_name}.csv'
            split_df.to_csv(file_path, index=False)
            self.logger.info(f"Saved {split_name} set to: {file_path}")
        
        # Save statistics
        stats_path = self.processed_dir / 'split_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save dataset info
        info = {
            'total_samples': sum([len(df) for df in splits.values()]),
            'splits': list(splits.keys()),
            'label_column': self.label_column,
            'text_column': self.text_column,
            'statistics': stats,
            'config': {
                'test_size': self.test_size,
                'val_size': self.val_size,
                'random_state': self.random_state
            }
        }
        
        info_path = self.processed_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Saved dataset info to: {info_path}")
    
    def load_splits(self):
        """
        Load previously saved splits
        
        Returns:
            Dictionary with splits
        """
        splits = {}
        
        for split_name in ['train', 'val', 'test']:
            file_path = self.processed_dir / f'{split_name}.csv'
            if file_path.exists():
                splits[split_name] = pd.read_csv(file_path)
                self.logger.info(f"Loaded {split_name} set: {len(splits[split_name])} samples")
            else:
                self.logger.warning(f"Split file not found: {file_path}")
                splits[split_name] = None
        
        return splits

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create DataLoader
    loader = DataLoader()
    
    # Load and process data
    df_raw = loader.load_raw_data()
    df_processed, stats = loader.preprocess_data(df_raw)
    splits, split_stats = loader.split_data(df_processed)