import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from data.dataset_loader import merge_datasets
from src.preprocessing import clean_text
from src.feature_extraction import extract_url_features

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_training_data():
    """
    Loads merged dataset using data.dataset_loader.merge_datasets().
    Returns a DataFrame with columns at minimum: ['email', 'label'].
    """
    df = merge_datasets()
    if "email" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'email' and 'label' columns")
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing: clean email text and add URL features (if url column exists).
    Returns a DataFrame with 'clean_email' column and numeric feature columns appended.
    """
    df = df.copy()
    df["clean_email"] = df["email"].astype(str).apply(clean_text)

    # If a 'url' column exists, extract URL features and merge them
    if "url" in df.columns:
        df = extract_url_features(df)  # this function returns df with new numeric url columns

    # Fill missing numeric columns with zeros (defensive)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def build_feature_matrix(df: pd.DataFrame, max_features: int = 2000):
    """
    Create TF-IDF features for 'clean_email' and concatenate numeric features (if any).
    Returns X (sparse matrix), y (Series), vectorizer (fitted TF-IDF).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df["clean_email"].astype(str))
    y = df["label"]

    # Find numeric columns other than label
    numeric_cols = [c for c in df.columns if c not in ("email", "clean_email", "label") and df[c].dtype.kind in "fiu"]
    if numeric_cols:
        import scipy.sparse as sp
        numeric_matrix = df[numeric_cols].astype(float).values
        X = sp.hstack((X_text, numeric_matrix))
    else:
        X = X_text

    return X, y, vectorizer


def train_model(df: pd.DataFrame):
    """
    Full training pipeline:
    - prepare df
    - build features
    - train RandomForest
    - save model & vectorizer
    Returns (model, vectorizer, X_test, y_test) for evaluation.
    """
    df_p = prepare_dataframe(df)
    X, y, vectorizer = build_feature_matrix(df_p)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # Save model & vectorizer
    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    return model, vectorizer, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Print classification report and accuracy.
    """
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))


if __name__ == "__main__":
    df = load_training_data()
    model, vectorizer, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)
    print("Training complete. Models saved to", MODELS_DIR)
