"""
Tests for data preprocessing
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from utils.text_cleaner import TextCleaner
from utils.data_loader import DataLoader

class TestTextCleaner(unittest.TestCase):
    """Test cases for TextCleaner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cleaner = TextCleaner()
        self.sample_text = "Hello! This is a test email. Visit https://example.com or email test@example.com. Call (123) 456-7890."
        
    def test_clean_text(self):
        """Test text cleaning"""
        cleaned = self.cleaner.clean_text(self.sample_text)
        self.assertIsInstance(cleaned, str)
        self.assertNotIn('https://', cleaned)
        self.assertNotIn('test@example.com', cleaned)
        self.assertNotIn('(123) 456-7890', cleaned)
        
    def test_clean_batch(self):
        """Test batch text cleaning"""
        texts = [self.sample_text, "Another test email", "Win free prize now!"]
        cleaned_texts = self.cleaner.clean_batch(texts, show_progress=False)
        self.assertEqual(len(cleaned_texts), len(texts))
        
    def test_cleaning_stats(self):
        """Test cleaning statistics"""
        cleaned = self.cleaner.clean_text(self.sample_text)
        stats = self.cleaner.get_cleaning_stats(self.sample_text, cleaned)
        self.assertIn('original_length', stats)
        self.assertIn('cleaned_length', stats)
        self.assertIn('reduction_percentage', stats)
        
    def test_html_cleaning(self):
        """Test HTML cleaning"""
        html_text = "<html><body>Test <b>email</b> with <a href='#'>link</a></body></html>"
        cleaner_with_html = TextCleaner(remove_html=True)
        cleaned = cleaner_with_html.clean_text(html_text)
        self.assertNotIn('<', cleaned)
        self.assertNotIn('>', cleaned)
        
    def test_stopword_removal(self):
        """Test stopword removal"""
        text_with_stopwords = "This is a test email with some stopwords"
        cleaner_with_stopwords = TextCleaner(remove_stopwords=True)
        cleaned = cleaner_with_stopwords.clean_text(text_with_stopwords)
        # Check that common stopwords are removed
        self.assertNotIn('this', cleaned)
        self.assertNotIn('is', cleaned)
        self.assertNotIn('a', cleaned)
        self.assertNotIn('with', cleaned)
        self.assertNotIn('some', cleaned)

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_loader = DataLoader()
        
        # Create a test DataFrame
        self.test_data = pd.DataFrame({
            'text': [
                'Test email 1',
                'Test email 2',
                'Test email 3',
                'Test email 4',
                'Test email 5'
            ],
            'spam': [0, 1, 0, 1, 0]
        })
        
    def test_preprocess_data(self):
        """Test data preprocessing"""
        df_processed, stats = self.data_loader.preprocess_data(self.test_data)
        
        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertIn('text_length', df_processed.columns)
        self.assertIn('word_count', df_processed.columns)
        self.assertIn('total_samples', stats)
        
    def test_split_data(self):
        """Test data splitting"""
        splits, split_stats = self.data_loader.split_data(self.test_data)
        
        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)
        
        total_len = len(splits['train']) + len(splits['val']) + len(splits['test'])
        self.assertEqual(total_len, len(self.test_data))
        
    def test_save_splits(self):
        """Test saving splits to files"""
        splits, split_stats = self.data_loader.split_data(self.test_data)
        
        # Check that files were created
        for split_name in ['train', 'val', 'test']:
            file_path = Path('data/processed') / f'{split_name}.csv'
            self.assertTrue(file_path.exists())
            
            # Clean up
            file_path.unlink(missing_ok=True)
            
    def test_load_splits(self):
        """Test loading splits from files"""
        # First save splits
        splits, split_stats = self.data_loader.split_data(self.test_data)
        
        # Then load them
        loaded_splits = self.data_loader.load_splits()
        
        self.assertIn('train', loaded_splits)
        self.assertIn('val', loaded_splits)
        self.assertIn('test', loaded_splits)
        
        # Clean up
        for split_name in ['train', 'val', 'test']:
            file_path = Path('data/processed') / f'{split_name}.csv'
            file_path.unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()