# preprocessing.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Common phishing keywords
phishing_keywords = ['urgent', 'act now', 'account suspended', 'limited offer', 'verify', 'confirm']


# Clean and normalize text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text


# Tokenize text
def tokenize_text(text):
    cleaned = clean_text(text)
    return word_tokenize(cleaned)


# Vectorize text using a passed-in vectorizer
def vectorize_text(text_series, vectorizer):
    return vectorizer.fit_transform(text_series)


# Extract features for a dataframe of emails
def extract_email_features(df):
    stop_words = set(stopwords.words('english'))

    # Use column "email" (matches your tests)
    df['cleaned_email'] = df['email'].apply(clean_text)

    df['contains_phishing_keywords'] = df['cleaned_email'].apply(
        lambda x: any(word in x for word in phishing_keywords)
    )

    df['email_length'] = df['cleaned_email'].apply(len)

    df['contains_stopwords'] = df['cleaned_email'].apply(
        lambda x: len([word for word in word_tokenize(x) if word in stop_words])
    )

    return df


# IMPORTANT:
# Do NOT run anything automatically on import.
# Only run example code when this file is executed directly.
if __name__ == "__main__":
    import pandas as pd

    # Example usage (OPTIONAL)
    df = pd.DataFrame({
        "email": [
            "URGENT! Your account has been suspended. Verify immediately.",
            "Let's meet tomorrow to discuss the project."
        ]
    })

    df = extract_email_features(df)
    print(df.head())
