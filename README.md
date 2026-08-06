# phishing-detection-system

Phase 1: Data Collection & Preprocessing
Gather a dataset of phishing and legitimate emails (e.g., from open datasets like PhishTank, OpenPhish, and Kaggle).
Extract key features:
Email Content Analysis: Use NLP to detect common phishing terms, misspellings, and urgency tactics.
URL Features: Check for domain age, length, HTTPS usage, and presence of known phishing patterns.
Sender Features: Analyze email headers, SPF, DKIM, and DMARC authentication.

Phase 2: Model Training
Preprocess text data using tokenization, stopword removal, and TF-IDF vectorization.
Train models:
Random Forest for feature-based classification.
Neural Networks (TensorFlow/PyTorch) for deep learning-based detection.

Phase 3: Web Scraping for Real-Time Detection
Use BeautifulSoup to analyze website content.
Check for deceptive UI elements and hidden links.

Phase 4: Deployment & Cloud Hosting
Develop an API using Flask or FastAPI for phishing detection.
Host the model on AWS, Google Cloud, or Azure.
Set up a web dashboard for monitoring detections.

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (terminal)
```bash
python -m src.model_training
```
This reads `data/phishing_emails.csv` and `data/legitimate_emails.csv`, trains a
TF-IDF + Random Forest classifier, and saves `models/random_forest.pkl` and
`models/tfidf_vectorizer.pkl`.

> **Note:** the bundled datasets are small samples (25 phishing / 25 legitimate
> emails) meant to demonstrate the pipeline end-to-end. For real-world accuracy,
> replace them with a much larger labeled dataset (e.g. from PhishTank, OpenPhish,
> or a Kaggle phishing-email dataset) as described in Phase 1 above, then re-run
> training.

### 3. Predict from the terminal
```bash
python -m src.predict "Your account has been suspended, click here to verify"
```

### 4. Predict from a browser (new)
```bash
python -m api.app
```
Then open **http://127.0.0.1:5000** in your browser, paste in an email, and
click "Scan email". This calls the same trained model behind a small Flask API
(`POST /api/predict`), so no terminal command is needed to use it day-to-day.
