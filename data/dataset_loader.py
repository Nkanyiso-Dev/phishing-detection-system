import pandas as pd
import os

def load_dataset(phishing_file, legitimate_file):
    """
    Load phishing and legitimate email datasets.
    
    :param phishing_file: Path to phishing emails CSV
    :param legitimate_file: Path to legitimate emails CSV
    :return: DataFrame containing both datasets with labels
    """
    
    if not os.path.exists(phishing_file):
        raise FileNotFoundError(f"Phishing dataset not found: {phishing_file}")

    if not os.path.exists(legitimate_file):
        raise FileNotFoundError(f"Legitimate dataset not found: {legitimate_file}")

    phishing_df = pd.read_csv(phishing_file)
    legitimate_df = pd.read_csv(legitimate_file)

    return pd.concat([phishing_df, legitimate_df], ignore_index=True)


def preprocess_dataset(dataset):
    """
    Perform basic preprocessing on the dataset like:
    - Removing NaN values
    - Removing duplicates
    - Resetting index
    
    :param dataset: Raw dataset
    :return: Cleaned dataset
    """
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()
    dataset = dataset.reset_index(drop=True)
    return dataset


def merge_datasets(
        phishing_file='data/phishing_emails.csv',
        legitimate_file='data/legitimate_emails.csv',
        shuffle=True,
        random_state=42
    ):
    """
    Load, clean, merge, and optionally shuffle phishing & legitimate datasets.
    
    :param phishing_file: Path to phishing emails CSV
    :param legitimate_file: Path to legitimate emails CSV
    :param shuffle: Whether to shuffle the combined dataset
    :param random_state: Seed for reproducible shuffling
    :return: Merged and preprocessed dataset
    """
    dataset = load_dataset(phishing_file, legitimate_file)
    dataset = preprocess_dataset(dataset)

    if shuffle:
        dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return dataset


if __name__ == '__main__':
    df = merge_datasets()
    print("\nMerged dataset preview:")
    print(df.head())
    print(f"\nTotal samples: {len(df)}")
