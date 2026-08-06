"""
app.py
------
Flask web application for the phishing detection system.

Serves:
  - GET  /                a simple browser UI for pasting in email text
  - POST /api/predict      JSON API: {"email": "..."} -> prediction result
  - GET  /api/health       basic health check

Run with:
    python -m api.app

Then open http://127.0.0.1:5000 in a browser.
"""

import os
import sys

# Allow running this file directly (python api/app.py) as well as via
# `python -m api.app` by making sure the project root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template

from src.predict import predict_email, ModelNotAvailableError

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)


@app.route("/")
def index():
    """Serve the browser UI."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    email_text = data.get("email", "")

    if not isinstance(email_text, str) or not email_text.strip():
        return jsonify({"error": "Please provide non-empty 'email' text."}), 400

    try:
        result = predict_email(email_text)
    except ModelNotAvailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


if __name__ == "__main__":
    # debug=True is convenient for local development; turn off for production.
    app.run(host="0.0.0.0", port=5000, debug=True)
