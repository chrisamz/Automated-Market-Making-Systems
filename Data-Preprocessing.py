# data_preprocessing.py

"""
Data Preprocessing Module for Automated Market-Making Systems

This module contains functions for collecting, cleaning, normalizing, and preparing market data for model training and evaluation.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction
- Handling missing data

Libraries/Tools:
- pandas
- numpy
- scikit-learn

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing duplicates and handling missing values.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.drop_duplicates()
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

        data[numeric_features] = self.numeric_imputer.fit_transform(data[numeric_features])
        return data

    def normalize_data(self, data):
        """
        Normalize the numeric data using standard scaling.
        
        :param data: DataFrame, input data
        :return: DataFrame, normalized data
        """
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        data[numeric_features] = self.scaler.fit_transform(data[numeric_features])
        return data

    def feature_extraction(self, data):
        """
        Extract relevant features from the data.
        
        :param data: DataFrame, input data
        :return: DataFrame, data with extracted features
        """
        # Example feature extraction logic (can be customized)
        data['price_diff'] = data['close'] - data['open']
        data['price_range'] = data['high'] - data['low']
        data['volume_log'] = np.log1p(data['volume'])
        return data

    def preprocess(self, raw_data_filepath, processed_data_dir):
        """
        Execute the full preprocessing pipeline.
        
        :param raw_data_filepath: str, path to the input data file
        :param processed_data_dir: str, directory to save processed data
        :return: DataFrame, preprocessed data
        """
        # Load data
        data = self.load_data(raw_data_filepath)

        # Clean data
        data = self.clean_data(data)

        # Extract features
        data = self.feature_extraction(data)

        # Normalize data
        data = self.normalize_data(data)

        # Save processed data
        os.makedirs(processed_data_dir, exist_ok=True)
        processed_data_filepath = os.path.join(processed_data_dir, 'processed_data.csv')
        data.to_csv(processed_data_filepath, index=False)
        print(f"Processed data saved to {processed_data_filepath}")

        return data

if __name__ == "__main__":
    raw_data_filepath = 'data/raw/market_data.csv'
    processed_data_dir = 'data/processed/'

    preprocessing = DataPreprocessing()

    # Preprocess the data
    processed_data = preprocessing.preprocess(raw_data_filepath, processed_data_dir)
    print("Data preprocessing completed and data saved.")
