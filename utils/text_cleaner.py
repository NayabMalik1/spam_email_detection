"""
Text preprocessing and cleaning utilities for email spam detection
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import html
from bs4 import BeautifulSoup
import unicodedata

class TextCleaner:
    """
    Comprehensive text cleaner for email preprocessing
    """
    
    def __init__(self, language='english', 
                 remove_stopwords=True,
                 stem_words=True,
                 lemmatize=False,
                 remove_html=True,
                 remove_urls=True,
                 remove_emails=True,
                 remove_phone_numbers=True,
                 remove_punctuation=True,
                 remove_numbers=True,
                 min_word_length=2):
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-eng', quiet=True)
        
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        self.lemmatize = lemmatize
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer() if stem_words else None
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
        # Compile regex patterns for better performance
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        
    def clean_text(self, text):
        """
        Apply all cleaning steps to the text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower().strip()
        
        # Remove HTML entities and tags
        if self.remove_html:
            text = html.unescape(text)
            text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Remove phone numbers
        if self.remove_phone_numbers:
            text = self.phone_pattern.sub('', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = self.punctuation_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if self.stem_words and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization (overrides stemming if both are True)
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter by minimum word length
        tokens = [token for token in tokens if len(token) >= self.min_word_length]
        
        # Join tokens back to text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def clean_batch(self, texts, show_progress=True):
        """
        Clean a batch of texts
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of cleaned texts
        """
        from tqdm import tqdm
        
        if show_progress:
            cleaned_texts = []
            for text in tqdm(texts, desc="Cleaning texts"):
                cleaned_texts.append(self.clean_text(text))
            return cleaned_texts
        else:
            return [self.clean_text(text) for text in texts]
    
    def get_cleaning_stats(self, original_text, cleaned_text):
        """
        Get statistics about the cleaning process
        
        Args:
            original_text: Original text
            cleaned_text: Cleaned text
            
        Returns:
            Dictionary with cleaning statistics
        """
        stats = {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'original_words': len(original_text.split()),
            'cleaned_words': len(cleaned_text.split()),
            'reduction_percentage': round((1 - len(cleaned_text)/len(original_text)) * 100, 2)
            if original_text else 0,
            'words_removed_percentage': round((1 - len(cleaned_text.split())/len(original_text.split())) * 100, 2)
            if original_text and original_text.split() else 0
        }
        return stats

# Example usage
if __name__ == "__main__":
    cleaner = TextCleaner()
    sample_text = "Hello! This is a test email. Visit https://example.com or email test@example.com. Call (123) 456-7890."
    cleaned = cleaner.clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned:", cleaned)
    print("Stats:", cleaner.get_cleaning_stats(sample_text, cleaned))