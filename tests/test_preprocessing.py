import unittest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import clean_text, tokenize_text, vectorize_text

class TestPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame({
            'email': [
                "Congratulations! You've won a free iPhone!!! Click here to claim now!",
                "Hi, can we schedule a meeting for tomorrow? Please confirm the time."
            ],
            'label': [1, 0]
        })
        cls.sample_email = cls.df['email'][0]

    def test_clean_text(self):
        cleaned_text = clean_text(self.sample_email)
        self.assertNotIn('!', cleaned_text)
        self.assertNotIn('?', cleaned_text)
        self.assertIn('congratulations', cleaned_text.lower())

    def test_tokenize_text(self):
        tokens = tokenize_text(self.sample_email)
        self.assertIsInstance(tokens, list)
        self.assertIn('congratulations', tokens)
        self.assertIn('won', tokens)

    def test_vectorize_text(self):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorize_text(self.df['email'], vectorizer)
        self.assertEqual(tfidf_matrix.shape[0], 2)
        self.assertGreater(tfidf_matrix.shape[1], 0)


if __name__ == '__main__':
    unittest.main()
