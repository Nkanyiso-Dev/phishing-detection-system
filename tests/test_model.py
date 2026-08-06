import os
import unittest
import tempfile
import shutil
import pandas as pd
import joblib
from unittest.mock import patch

# Import the functions we expect in src.model_training
from src import model_training


class TestModelTrainingPipeline(unittest.TestCase):
    def setUp(self):
        # create a temporary models dir to avoid touching real models
        self.original_models_dir = model_training.MODELS_DIR
        self.tmpdir = tempfile.mkdtemp(prefix="models_test_")
        model_training.MODELS_DIR = self.tmpdir

        # create a small mock dataset
        self.mock_df = pd.DataFrame({
            "email": [
                "URGENT: Your account has been suspended. Click here to verify.",
                "Hello team, meeting at 3pm regarding project status.",
                "Win a free iPhone by clicking this link now!",
                "Please find the attached report for last quarter."
            ],
            "label": [1, 0, 1, 0],
            # include url column to exercise extract_url_features branch if available
            "url": [
                "http://phish.example/verify",
                "https://intranet.company.local/report",
                "http://short.ly/free",
                ""
            ]
        })

    def tearDown(self):
        # restore and cleanup
        model_training.MODELS_DIR = self.original_models_dir
        shutil.rmtree(self.tmpdir)

        # remove any created files (defensive)
        for fname in ("random_forest.pkl", "tfidf_vectorizer.pkl"):
            path = os.path.join(self.tmpdir, fname)
            if os.path.exists(path):
                os.remove(path)

    @patch("data.dataset_loader.merge_datasets")
    def test_train_and_save_model(self, mock_merge):
        # Patch merge_datasets to return our mock_df
        mock_merge.return_value = self.mock_df.copy()

        # Run training
        model, vectorizer, X_test, y_test = model_training.train_model(model_training.load_training_data())

        # Check returned objects
        self.assertIsNotNone(model)
        self.assertIsNotNone(vectorizer)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_test)

        # Check that files were saved
        model_path = os.path.join(model_training.MODELS_DIR, "random_forest.pkl")
        vec_path = os.path.join(model_training.MODELS_DIR, "tfidf_vectorizer.pkl")
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(vec_path))

        # Load the saved model to ensure it's a valid joblib object
        loaded = joblib.load(model_path)
        self.assertTrue(hasattr(loaded, "predict"))

    @patch("data.dataset_loader.merge_datasets")
    def test_prepare_dataframe_outputs_clean_email(self, mock_merge):
        mock_merge.return_value = self.mock_df.copy()
        df = model_training.load_training_data()
        prepared = model_training.prepare_dataframe(df)
        self.assertIn("clean_email", prepared.columns)
        # pandas >= 2.x may report a dedicated "str" extension dtype instead of
        # plain "object" for string columns depending on configuration; both
        # are valid, what matters is the values are actually strings.
        self.assertTrue(
            prepared["clean_email"].dtype == object
            or prepared["clean_email"].dtype.kind in ("O", "U")
            or all(isinstance(v, str) for v in prepared["clean_email"])
        )

    @patch("data.dataset_loader.merge_datasets")
    def test_build_feature_matrix_shapes(self, mock_merge):
        mock_merge.return_value = self.mock_df.copy()
        df = model_training.load_training_data()
        prepared = model_training.prepare_dataframe(df)
        X, y, vectorizer = model_training.build_feature_matrix(prepared, max_features=50)
        # X should have same number of rows as df
        self.assertEqual(X.shape[0], prepared.shape[0])
        self.assertEqual(len(y), prepared.shape[0])
        self.assertIsNotNone(vectorizer)


if __name__ == "__main__":
    unittest.main()
