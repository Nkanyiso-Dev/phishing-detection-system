"""
predict.py
----------
Inference layer for the phishing detection system.

Loads the trained TF-IDF vectorizer + RandomForest model (produced by
src/model_training.py) and exposes a simple API for scoring raw email text.

Usable two ways:
  1. As a library:  from src.predict import predict_email
  2. As a CLI:       python -m src.predict "some email text"
                      python -m src.predict --file path/to/email.txt
"""

import os
import sys
import argparse
import joblib

from src.preprocessing import clean_text

# Resolve paths relative to the project root (not the current working
# directory), so this works no matter where it's invoked from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

_model = None
_vectorizer = None


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained model/vectorizer files can't be found or loaded."""


def load_artifacts():
    """
    Load (and cache) the trained model and vectorizer from disk.
    Safe to call repeatedly - only loads from disk once.
    """
    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise ModelNotAvailableError(
            "Trained model files not found. Run 'python -m src.model_training' "
            f"first to generate them (expected at {MODEL_PATH} and {VECTORIZER_PATH})."
        )

    try:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as exc:
        raise ModelNotAvailableError(f"Failed to load model artifacts: {exc}") from exc

    return _model, _vectorizer


def predict_email(email_text: str) -> dict:
    """
    Score a single piece of email text.

    Returns a dict:
        {
            "is_phishing": bool,
            "label": 0 or 1,
            "confidence": float (0-1, probability of the predicted class),
            "phishing_probability": float (0-1),
        }
    """
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("email_text must be a non-empty string")

    model, vectorizer = load_artifacts()

    cleaned = clean_text(email_text)
    X = vectorizer.transform([cleaned])

    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    # model.classes_ is [0, 1] -> index 1 is the "phishing" probability
    class_index = {c: i for i, c in enumerate(model.classes_)}
    phishing_probability = float(proba[class_index.get(1, 1)])
    confidence = float(proba[class_index.get(label, label)])

    return {
        "is_phishing": bool(label == 1),
        "label": label,
        "confidence": round(confidence, 4),
        "phishing_probability": round(phishing_probability, 4),
    }


def _main():
    parser = argparse.ArgumentParser(description="Predict whether an email is phishing.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("text", nargs="?", help="Raw email text to classify")
    group.add_argument("--file", "-f", help="Path to a text file containing the email")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            email_text = fh.read()
    else:
        email_text = args.text

    try:
        result = predict_email(email_text)
    except ModelNotAvailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    verdict = "PHISHING" if result["is_phishing"] else "LEGITIMATE"
    print(f"Verdict: {verdict}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Phishing probability: {result['phishing_probability'] * 100:.1f}%")


if __name__ == "__main__":
    _main()
