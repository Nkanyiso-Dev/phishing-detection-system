# feature_extraction.py
from urllib.parse import urlparse
import requests
import tldextract
from datetime import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup

def extract_url_features(df):
    # Function to extract features from URL
    def extract_features(url):
        features = {}
        try:
            parsed_url = urlparse(url)
            ext = tldextract.extract(url)
            features['domain'] = ext.domain
            features['subdomain'] = ext.subdomain
            features['path_length'] = len(parsed_url.path)
            features['is_https'] = 1 if parsed_url.scheme == 'https' else 0
            features['url_length'] = len(url)
        except Exception as e:
            features = {'domain': '', 'subdomain': '', 'path_length': 0, 'is_https': 0, 'url_length': 0}
        return features
    
    # Apply URL feature extraction
    url_features = df['url'].apply(extract_features)
    url_df = pd.DataFrame(url_features.tolist())
    
    # Concatenate URL features with the main dataframe
    df = pd.concat([df, url_df], axis=1)
    return df

# Example usage:
df = extract_url_features(df)
print(df[['url', 'domain', 'is_https', 'url_length']].head())
