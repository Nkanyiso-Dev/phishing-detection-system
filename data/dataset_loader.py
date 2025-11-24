# dataset_loader.py
import pandas as pd

def load_dataset(phishing_file, legitimate_file):
    """
    Load phishing and legitimate email datasets.
    
    :param phishing_file: Path to phishing emails CSV
    :param legitimate_file: Path to legitimate emails CSV
    :return: DataFrame containing both datasets with labels
    """
    phishing_df = pd.read_csv(phishing_file)
    legitimate_df = pd.read_csv(legitimate_file)
    
    # Concatenate both datasets
    dataset = pd.concat([phishing_df, legitimate_df], ignore_index=True)
    return dataset

def preprocess_dataset(dataset):
    """
    Perform basic preprocessing on the dataset like removing NaN values
    and resetting index.
    
    :param dataset: Raw dataset
    :return: Cleaned dataset
    """
    dataset = dataset.dropna()  # Drop any rows with NaN values
    dataset = dataset.reset_index(drop=True)
    return dataset

if __name__ == '__main__':
    phishing_file = 'data/phishing_emails.csv'
    legitimate_file = 'data/legitimate_emails.csv'
    
    # Load and preprocess dataset
    data = load_dataset(phishing_file, legitimate_file)
    clean_data = preprocess_dataset(data)
    
    print(clean_data.head())
