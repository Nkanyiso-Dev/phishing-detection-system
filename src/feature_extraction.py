import re
import pandas as pd
from urllib.parse import urlparse
import tldextract
import math


# ---------------------------------------------------------
# Helper: Calculate entropy of a string (useful for obfuscated URLs)
# ---------------------------------------------------------
def calculate_entropy(text):
    if not text:
        return 0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum([p * math.log(p, 2) for p in prob])
    return entropy


# ---------------------------------------------------------
# Helper: Check if URL uses an IP instead of domain
# ---------------------------------------------------------
def url_contains_ip(url):
    ip_pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
    return 1 if re.search(ip_pattern, url) else 0


# ---------------------------------------------------------
# Helper: Detect shortening services
# ---------------------------------------------------------
SHORTENER_DOMAINS = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "bit.do",
    "shorte.st", "adf.ly", "soo.gd", "is.gd", "buff.ly"
]


def is_shortened_url(url):
    try:
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        return 1 if domain in SHORTENER_DOMAINS else 0
    except:
        return 0


# ---------------------------------------------------------
# Main URL feature extractor
# ---------------------------------------------------------
def extract_url_features(df):
    """
    Extract multiple URL-based phishing features.
    """

    def extract(url):
        # Default values
        features = {
            "url_length": 0,
            "path_length": 0,
            "num_dots": 0,
            "is_https": 0,
            "is_ip": 0,
            "num_params": 0,
            "entropy": 0,
            "suspicious_keywords": 0,
            "is_shortened": 0,
            "tld_length": 0
        }

        if not isinstance(url, str) or url.strip() == "":
            return features

        try:
            parsed = urlparse(url)
            ext = tldextract.extract(url)

            features["url_length"] = len(url)
            features["path_length"] = len(parsed.path)
            features["num_dots"] = url.count(".")
            features["is_https"] = 1 if parsed.scheme == "https" else 0
            features["num_params"] = url.count("=")
            features["entropy"] = calculate_entropy(url)
            features["tld_length"] = len(ext.suffix or "")

            # Check if URL uses IP instead of domain name
            features["is_ip"] = url_contains_ip(url)

            # Detect shortening service
            features["is_shortened"] = is_shortened_url(url)

            # Suspicious keyword detection
            sus_words = [
                "verify", "update", "secure", "login", "confirm",
                "free", "winner", "click", "urgent", "bank",
                "password", "reset", "account"
            ]
            features["suspicious_keywords"] = sum(1 for w in sus_words if w in url.lower())

        except Exception:
            pass

        return features

    # Apply the extractor
    url_features = df["url"].apply(extract)
    url_df = pd.DataFrame(list(url_features))

    return pd.concat([df, url_df], axis=1)
