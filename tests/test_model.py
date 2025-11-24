import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import load_model
import os

class TestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load dataset and split
        cls.df = pd.read_csv('data/phishing_emails.csv')  # Ensure this file exists
        cls.X = cls.df.drop(columns=['label'])
        cls.y = cls.df['label']
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

        # Train/load models
        cls.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        cls.rf_model.fit(cls.X_train, cls.y_train)
        
        # Load the neural network model
        cls.nn_model = load_model('models/neural_network.h5')  # Ensure this file exists

    def test_random_forest(self):
        y_pred_rf = self.rf_model.predict(self.X_test)
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf)
        self.assertGreater(accuracy_rf, 0.8, f"Random Forest accuracy too low: {accuracy_rf}")

    def test_neural_network(self):
        y_pred_nn = self.nn_model.predict(self.X_test)
        y_pred_nn = (y_pred_nn > 0.5).astype(int)  # Convert output to binary (0 or 1)
        accuracy_nn = accuracy_score(self.y_test, y_pred_nn)
        self.assertGreater(accuracy_nn, 0.8, f"Neural Network accuracy too low: {accuracy_nn}")


if __name__ == '__main__':
    unittest.main()