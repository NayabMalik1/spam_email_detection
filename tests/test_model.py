"""
Tests for the ANN model
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path

from models.ann_model import ANNModel
from models.text_vectorizer import TextVectorizer

class TestANNModel(unittest.TestCase):
    """Test cases for ANNModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ann_model = ANNModel()
        self.n_samples = 100
        self.n_features = 100
        
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.randint(0, 2, self.n_samples)
        
    def test_build_model(self):
        """Test model building"""
        model = self.ann_model.build_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape[1], self.ann_model.input_dim)
        
    def test_model_summary(self):
        """Test model summary generation"""
        self.ann_model.build_model()
        # Just check that it doesn't crash
        self.assertIsNotNone(self.ann_model.model.summary())
        
    def test_train_model(self):
        """Test model training (short version)"""
        self.ann_model.epochs = 2  # Reduced for testing
        self.ann_model.build_model()
        
        history = self.ann_model.train(self.X_train, self.y_train)
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        
    def test_save_load_model(self):
        """Test model saving and loading"""
        self.ann_model.build_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model.h5'
            saved_path = self.ann_model.save_model(model_path)
            self.assertTrue(Path(saved_path).exists())
            
            # Load model
            loaded_model = self.ann_model.load_model(saved_path)
            self.assertIsNotNone(loaded_model)
            
    def test_predict(self):
        """Test model predictions"""
        self.ann_model.build_model()
        
        # Train briefly
        self.ann_model.epochs = 1
        self.ann_model.train(self.X_train, self.y_train)
        
        # Make predictions
        probabilities, predictions = self.ann_model.predict(self.X_train[:10])
        self.assertEqual(len(probabilities), 10)
        self.assertEqual(len(predictions), 10)
        
    def test_predict_single(self):
        """Test single prediction"""
        self.ann_model.build_model()
        
        # Train briefly
        self.ann_model.epochs = 1
        self.ann_model.train(self.X_train, self.y_train)
        
        # Make single prediction
        result = self.ann_model.predict_single(self.X_train[0])
        self.assertIn('probability', result)
        self.assertIn('prediction', result)
        self.assertIn('class', result)

class TestTextVectorizer(unittest.TestCase):
    """Test cases for TextVectorizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vectorizer = TextVectorizer()
        self.texts = [
            "This is a test email about machine learning",
            "Win money now click here free prize",
            "Meeting scheduled for tomorrow afternoon",
            "Buy cheap viagra online pharmacy discount"
        ]
        
    def test_fit_transform_tfidf(self):
        """Test TF-IDF vectorization"""
        X = self.vectorizer.fit_transform(self.texts, method='tfidf')
        self.assertEqual(X.shape[0], len(self.texts))
        self.assertGreater(X.shape[1], 0)
        
    def test_fit_transform_count(self):
        """Test Count vectorization"""
        X = self.vectorizer.fit_transform(self.texts, method='count')
        self.assertEqual(X.shape[0], len(self.texts))
        self.assertGreater(X.shape[1], 0)
        
    def test_save_load_vectorizer(self):
        """Test vectorizer saving and loading"""
        self.vectorizer.fit(self.texts, method='tfidf')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vectorizer_path = Path(tmpdir) / 'test_vectorizer.pkl'
            saved_path = self.vectorizer.save_vectorizer(vectorizer_path)
            self.assertTrue(Path(saved_path).exists())
            
            # Load vectorizer
            loaded_vectorizer = TextVectorizer()
            loaded_vectorizer.load_vectorizer(saved_path)
            self.assertIsNotNone(loaded_vectorizer.vectorizer)
            
    def test_feature_importance(self):
        """Test feature importance calculation"""
        self.vectorizer.fit(self.texts, method='tfidf')
        X = self.vectorizer.transform(self.texts)
        
        df_importance = self.vectorizer.get_feature_importance(X=X)
        self.assertGreater(len(df_importance), 0)
        self.assertIn('feature', df_importance.columns)
        self.assertIn('importance', df_importance.columns)

if __name__ == '__main__':
    unittest.main()