"""
Text vectorization using TF-IDF and other methods
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from pathlib import Path
import json
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

import yaml
from pathlib import Path

class TextVectorizer:
    def __init__(self, config_path=None):
        if config_path is None:
            # Get absolute path to configs/config.yaml from project root
            config_path = Path(__file__).resolve().parent.parent / 'configs' / 'config.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Vectorizer parameters
        self.max_features = self.config['data']['max_features']
        self.method = 'tfidf'  # Default method
        
        # Initialize vectorizers
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.vectorizer = None
        
        # Paths
        self.models_dir = Path(self.config['paths']['models'])
        self.plots_dir = Path(self.config['paths']['plots'])
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.vocabulary_ = None
        self.feature_names_ = None
        self.vectorizer_stats = {}
    
    def fit(self, texts: List[str], method: str = 'tfidf'):
        """
        Fit vectorizer on texts
        
        Args:
            texts: List of text strings
            method: Vectorization method ('tfidf' or 'count')
            
        Returns:
            Fitted vectorizer
        """
        self.logger.info(f"Fitting {method.upper()} vectorizer...")
        self.logger.info(f"Number of documents: {len(texts)}")
        self.logger.info(f"Max features: {self.max_features}")
        
        self.method = method.lower()
        
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2),  # Use unigrams and bigrams
                stop_words='english',
                sublinear_tf=True,  # Use 1 + log(tf)
                smooth_idf=True,
                norm='l2'
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tfidf' or 'count'")
        
        # Fit vectorizer
        self.vectorizer.fit(texts)
        
        # Store vocabulary and feature names
        self.vocabulary_ = self.vectorizer.vocabulary_
        self.feature_names_ = self.vectorizer.get_feature_names_out()
        
        # TEMPORARY FIX: Skip statistics calculation for now
        # self._calculate_statistics(texts)
        
        # Simple logging instead
        X = self.transform(texts)
        self.logger.info(f"Vectorizer fitted successfully")
        self.logger.info(f"Vocabulary size: {len(self.vocabulary_)}")
        self.logger.info(f"Feature matrix shape: {X.shape}")
        
        return self.vectorizer
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to vectors
        
        Args:
            texts: List of text strings
            
        Returns:
            Vectorized texts as numpy array
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str], method: str = 'tfidf') -> np.ndarray:
        """
        Fit vectorizer and transform texts
        
        Args:
            texts: List of text strings
            method: Vectorization method
            
        Returns:
            Vectorized texts as numpy array
        """
        self.fit(texts, method)
        return self.transform(texts)
    
    def _calculate_statistics(self, texts: List[str]):
        """
        Calculate statistics about the vectorized texts
        
        Args:
            texts: List of text strings
        """
        # Transform texts
        X = self.transform(texts)
        
        # For sparse matrices, use different methods
        if hasattr(X, 'count_nonzero'):
            # Sparse matrix case
            non_zero_count = X.count_nonzero()
            nonzero_features_per_doc = np.diff(X.indptr)  # Count non-zero per document
        else:
            # Dense array case
            non_zero_count = np.count_nonzero(X)
            nonzero_features_per_doc = np.count_nonzero(X, axis=1)
        
        # FIXED: Calculate total elements (outside if-else block)
        total_elements = X.shape[0] * X.shape[1]
        
        # Calculate statistics
        self.vectorizer_stats = {
            'method': self.method,
            'n_documents': len(texts),
            'n_features': X.shape[1],
            'sparsity': 1.0 - (non_zero_count / total_elements) if total_elements > 0 else 0.0,
            'avg_nonzero_features': float(np.mean(nonzero_features_per_doc)) if len(nonzero_features_per_doc) > 0 else 0.0,
            'total_vocabulary_size': len(self.vocabulary_),
            'feature_density': non_zero_count / total_elements if total_elements > 0 else 0.0
        }
        
        # Calculate feature importance if using TF-IDF
        if self.method == 'tfidf':
            feature_importance = np.array(X.sum(axis=0)).flatten()
            top_indices = np.argsort(feature_importance)[-20:]  # Top 20 features
            self.vectorizer_stats['top_features'] = {
                self.feature_names_[i]: float(feature_importance[i])
                for i in top_indices[::-1]  # Reverse to get highest first
            }
        
        self.logger.info("Vectorizer statistics:")
        for key, value in self.vectorizer_stats.items():
            if key != 'top_features':
                self.logger.info(f"  {key}: {value}")
    
    def plot_feature_distribution(self, X: np.ndarray = None, texts: List[str] = None, 
                                 save: bool = True):
        """
        Plot distribution of features
        
        Args:
            X: Vectorized texts (optional)
            texts: Raw texts (optional)
            save: Whether to save the plot
            
        Returns:
            matplotlib figure
        """
        if X is None and texts is not None:
            X = self.transform(texts)
        elif X is None:
            raise ValueError("Either X or texts must be provided")
        
        # Calculate feature frequencies
        feature_frequencies = np.array(X.sum(axis=0)).flatten()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flat
        
        # 1. Distribution of feature frequencies
        ax = axes[0]
        ax.hist(feature_frequencies[feature_frequencies > 0], bins=50, 
               alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Feature Frequency')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Feature Frequencies')
        ax.grid(True, alpha=0.3)
        
        # 2. Top features bar chart
        ax = axes[1]
        top_n = 20
        top_indices = np.argsort(feature_frequencies)[-top_n:]
        top_features = [self.feature_names_[i] for i in top_indices]
        top_values = feature_frequencies[top_indices]
        
        bars = ax.barh(range(top_n), top_values[::-1], 
                      color=plt.cm.tab20c(np.arange(top_n)))
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_n} Most Frequent Features')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Sparsity visualization
        ax = axes[2]
        nonzero_per_doc = np.count_nonzero(X, axis=1)
        ax.hist(nonzero_per_doc, bins=50, alpha=0.7, 
               color='lightgreen', edgecolor='black')
        ax.set_xlabel('Number of Non-zero Features per Document')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Non-zero Features per Document')
        ax.grid(True, alpha=0.3)
        
        # 4. Feature correlation heatmap (top features only)
        ax = axes[3]
        if len(top_indices) > 1:
            top_features_matrix = X[:, top_indices[-10:]]  # Top 10 features
            corr_matrix = np.corrcoef(top_features_matrix.T)
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', 
                          vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xticklabels([self.feature_names_[i] for i in top_indices[-10:]], 
                              rotation=45, ha='right')
            ax.set_yticklabels([self.feature_names_[i] for i in top_indices[-10:]])
            ax.set_title('Correlation Matrix of Top 10 Features')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'{self.method.upper()} Vectorizer Feature Analysis', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.plots_dir / f'feature_distribution_{self.method}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature distribution plot saved to: {save_path}")
        
        return fig
    
    def create_word_cloud(self, texts: List[str], labels: np.ndarray = None, 
                         save: bool = True):
        """
        Create word clouds for spam and ham emails
        
        Args:
            texts: List of text strings
            labels: Binary labels (0 for ham, 1 for spam)
            save: Whether to save the plots
            
        Returns:
            List of word cloud figures
        """
        if labels is None:
            # Create single word cloud for all texts
            all_text = ' '.join(texts)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=200,
                contour_width=3,
                contour_color='steelblue'
            ).generate(all_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud - All Emails')
            
            if save:
                save_path = self.plots_dir / 'wordcloud_all.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Word cloud saved to: {save_path}")
            
            return [fig]
        
        else:
            # Separate spam and ham texts
            spam_texts = [text for text, label in zip(texts, labels) if label == 1]
            ham_texts = [text for text, label in zip(texts, labels) if label == 0]
            
            spam_text = ' '.join(spam_texts) if spam_texts else ''
            ham_text = ' '.join(ham_texts) if ham_texts else ''
            
            figs = []
            
            # Spam word cloud
            if spam_text:
                wordcloud_spam = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=200,
                    contour_width=3,
                    contour_color='red'
                ).generate(spam_text)
                
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.imshow(wordcloud_spam, interpolation='bilinear')
                ax1.axis('off')
                ax1.set_title('Word Cloud - Spam Emails')
                figs.append(fig1)
                
                if save:
                    save_path = self.plots_dir / 'wordcloud_spam.png'
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Ham word cloud
            if ham_text:
                wordcloud_ham = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=200,
                    contour_width=3,
                    contour_color='green'
                ).generate(ham_text)
                
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.imshow(wordcloud_ham, interpolation='bilinear')
                ax2.axis('off')
                ax2.set_title('Word Cloud - Ham Emails')
                figs.append(fig2)
                
                if save:
                    save_path = self.plots_dir / 'wordcloud_ham.png'
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            self.logger.info("Word clouds created successfully")
            return figs
    
    def reduce_dimensionality(self, X: np.ndarray, n_components: int = 2, 
                             method: str = 'svd'):
        """
        Reduce dimensionality for visualization
        
        Args:
            X: Vectorized texts
            n_components: Number of components
            method: Dimensionality reduction method ('svd' or 'lda')
            
        Returns:
            Reduced features
        """
        self.logger.info(f"Reducing dimensionality using {method.upper()}...")
        
        if method == 'svd':
            reducer = TruncatedSVD(
                n_components=n_components,
                random_state=42
            )
        elif method == 'lda':
            reducer = LatentDirichletAllocation(
                n_components=n_components,
                random_state=42,
                max_iter=10
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'svd' or 'lda'")
        
        X_reduced = reducer.fit_transform(X)
        
        self.logger.info(f"Reduced from {X.shape[1]} to {n_components} dimensions")
        
        return X_reduced, reducer
    
    def save_vectorizer(self, path: str = None):
        """
        Save vectorizer to disk
        
        Args:
            path: Path to save vectorizer
            
        Returns:
            Path where vectorizer was saved
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        if path is None:
            path = self.models_dir / f'vectorizer_{self.method}.pkl'
        
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'method': self.method,
                'vocabulary': self.vocabulary_,
                'feature_names': self.feature_names_,
                'stats': self.vectorizer_stats
            }, f)
        
        # Save statistics as JSON
        stats_path = path.with_suffix('.json')
        with open(stats_path, 'w') as f:
            json.dump(self.vectorizer_stats, f, indent=2)
        
        self.logger.info(f"Vectorizer saved to: {path}")
        self.logger.info(f"Vectorizer statistics saved to: {stats_path}")
        
        return path
    
    def load_vectorizer(self, path: str):
        """
        Load vectorizer from disk
        
        Args:
            path: Path to saved vectorizer
            
        Returns:
            Loaded vectorizer
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Vectorizer file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.method = data['method']
        self.vocabulary_ = data['vocabulary']
        self.feature_names_ = data['feature_names']
        self.vectorizer_stats = data['stats']
        
        self.logger.info(f"Vectorizer loaded from: {path}")
        self.logger.info(f"Method: {self.method}")
        self.logger.info(f"Vocabulary size: {len(self.vocabulary_)}")
        
        return self.vectorizer
    
    def get_feature_importance(self, X: np.ndarray = None, 
                              texts: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Args:
            X: Vectorized texts (optional)
            texts: Raw texts (optional)
            
        Returns:
            DataFrame with feature importance
        """
        if X is None and texts is not None:
            X = self.transform(texts)
        elif X is None:
            raise ValueError("Either X or texts must be provided")
        
        # Calculate feature importance (mean TF-IDF score)
        feature_importance = np.array(X.mean(axis=0)).flatten()
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': feature_importance,
            'frequency': np.array(X.sum(axis=0)).flatten()
        })
        
        # Sort by importance
        df_importance = df_importance.sort_values('importance', ascending=False)
        
        return df_importance
    
    def analyze_ngrams(self, texts: List[str], n: int = 2, top_n: int = 20):
        """
        Analyze n-grams in texts
        
        Args:
            texts: List of text strings
            n: N-gram size
            top_n: Number of top n-grams to return
            
        Returns:
            DataFrame with n-gram analysis
        """
        # Create n-gram vectorizer
        ngram_vectorizer = CountVectorizer(
            ngram_range=(n, n),
            max_features=1000,
            stop_words='english'
        )
        
        # Fit and transform
        X_ngram = ngram_vectorizer.fit_transform(texts)
        
        # Get feature names and frequencies
        feature_names = ngram_vectorizer.get_feature_names_out()
        frequencies = np.array(X_ngram.sum(axis=0)).flatten()
        
        # Create DataFrame
        df_ngrams = pd.DataFrame({
            'ngram': feature_names,
            'frequency': frequencies
        })
        
        # Sort and get top n-grams
        df_ngrams = df_ngrams.sort_values('frequency', ascending=False)
        df_ngrams = df_ngrams.head(top_n)
        
        # Plot top n-grams
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(df_ngrams)), df_ngrams['frequency'][::-1])
        ax.set_yticks(range(len(df_ngrams)))
        ax.set_yticklabels(df_ngrams['ngram'][::-1])
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_n} {n}-grams')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.plots_dir / f'top_{n}_grams.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"N-gram analysis plot saved to: {save_path}")
        
        return df_ngrams, fig

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    texts = [
        "Hello this is a test email about machine learning",
        "Win money now click here free prize",
        "Meeting scheduled for tomorrow afternoon",
        "Buy cheap viagra online pharmacy discount",
        "Project update and status report"
    ]
    
    labels = np.array([0, 1, 0, 1, 0])  # 0 = ham, 1 = spam
    
    # Initialize vectorizer
    vectorizer = TextVectorizer()
    
    # Fit and transform
    X = vectorizer.fit_transform(texts, method='tfidf')
    
    # Plot feature distribution
    vectorizer.plot_feature_distribution(X=X)
    
    # Create word clouds
    vectorizer.create_word_cloud(texts, labels)
    
    # Get feature importance
    df_importance = vectorizer.get_feature_importance(X=X)
    print("\nTop 10 features by importance:")
    print(df_importance.head(10))
    
    # Analyze bigrams
    df_bigrams, fig = vectorizer.analyze_ngrams(texts, n=2, top_n=15)
    
    # Save vectorizer
    vectorizer.save_vectorizer()