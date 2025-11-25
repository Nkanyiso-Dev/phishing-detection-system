import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

PHISHING_WORDS = [
    "urgent", "verify", "bank", "password", "reset", "security", "account",
    "confirm", "winner", "free", "money", "alert", "click", "compromised",
    "suspended", "limited", "offer"
]

def extract_custom_features(text):
    text_lower = text.lower()

    return {
        "length": len(text),
        "exclamation_count": text.count("!"),
        "uppercase_words": sum(1 for w in text.split() if w.isupper()),
        "phishing_keyword_count": sum(1 for word in PHISHING_WORDS if word in text_lower)
    }

# Load data
df = pd.read_csv("phishing_emails.csv")

# Create feature dataframe
custom_features = df["email"].apply(extract_custom_features).apply(pd.Series)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_text = vectorizer.fit_transform(df["email"])
X_all = pd.concat([custom_features, pd.DataFrame(X_text.toarray())], axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Model A: Logistic Regression
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n=== Logistic Regression ===")
print(classification_report(y_test, lr_pred))

# Model B: Random Forest
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print(classification_report(y_test, rf_pred))

# Choose best (here we select LR for small datasets)
best_model = lr_model

# Save model + vectorizer
joblib.dump(best_model, "models/phishing_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel training complete. Model saved in /models/")
