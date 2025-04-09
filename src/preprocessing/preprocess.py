import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
        self.processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize preprocessing objects
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
        # Define financial ratios and features
        self.financial_ratios = [
            'current_ratio',
            'debt_to_equity',
            'return_on_assets',
            'return_on_equity',
            'profit_margin',
            'asset_turnover'
        ]

    def load_data(self):
        """Load all raw data files"""
        logger.info("Loading raw data...")
        
        # Load companies data
        companies_df = pd.read_csv(os.path.join(self.raw_data_dir, 'companies.csv'))
        
        # Load financial statements for all companies
        financial_dfs = []
        for file in os.listdir(self.raw_data_dir):
            if file.startswith('financial_statements_') and file.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.raw_data_dir, file))
                financial_dfs.append(df)
        
        financial_df = pd.concat(financial_dfs, ignore_index=True)
        
        # Load news data
        news_dfs = []
        for file in os.listdir(self.raw_data_dir):
            if file.startswith('news_') and file.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.raw_data_dir, file))
                news_dfs.append(df)
        
        news_df = pd.concat(news_dfs, ignore_index=True)
        
        return companies_df, financial_df, news_df

    def clean_financial_data(self, df):
        """Clean and preprocess financial data"""
        logger.info("Cleaning financial data...")
        
        # Convert date columns to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns, removing currency symbols and commas
        numeric_columns = ['revenue', 'profit', 'assets', 'liabilities']
        for col in numeric_columns:
            df[col] = df[col].str.replace(',', '').str.replace('₫', '').astype(float)
        
        # Calculate financial ratios
        df['current_ratio'] = df['assets'] / df['liabilities']
        df['debt_to_equity'] = df['liabilities'] / (df['assets'] - df['liabilities'])
        df['return_on_assets'] = df['profit'] / df['assets']
        df['return_on_equity'] = df['profit'] / (df['assets'] - df['liabilities'])
        df['profit_margin'] = df['profit'] / df['revenue']
        df['asset_turnover'] = df['revenue'] / df['assets']
        
        # Handle infinite values and NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df

    def process_news_data(self, df):
        """Process and analyze news data"""
        logger.info("Processing news data...")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Add sentiment analysis features (placeholder)
        # In a real implementation, you would use a proper sentiment analysis model
        df['sentiment_score'] = np.random.normal(0, 1, len(df))  # Placeholder
        
        # Group by company and calculate news statistics
        news_stats = df.groupby('symbol').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        news_stats.columns = ['symbol'] + [f'news_{col[0]}_{col[1]}' for col in news_stats.columns[1:]]
        
        return news_stats

    def create_features(self, financial_df, news_stats):
        """Create final feature set for modeling"""
        logger.info("Creating final feature set...")
        
        # Merge financial and news data
        features_df = financial_df.merge(news_stats, on='symbol', how='left')
        
        # Select features for modeling
        feature_columns = self.financial_ratios + [col for col in features_df.columns if col.startswith('news_')]
        
        # Scale features
        features_df[feature_columns] = self.scaler.fit_transform(features_df[feature_columns])
        
        return features_df

    def prepare_training_data(self, features_df):
        """Prepare data for training"""
        logger.info("Preparing training data...")
        
        # Create labels for business health (example criteria)
        features_df['is_healthy'] = (
            (features_df['current_ratio'] > 1.5) &
            (features_df['debt_to_equity'] < 2) &
            (features_df['return_on_assets'] > 0.05) &
            (features_df['profit_margin'] > 0.1)
        ).astype(int)
        
        # Create labels for fraud detection (example criteria)
        features_df['is_fraud'] = (
            (features_df['profit_margin'] > 0.5) |  # Unrealistically high profit margin
            (features_df['asset_turnover'] > 10) |  # Unrealistically high asset turnover
            (features_df['news_sentiment_score_std'] > 2)  # High volatility in news sentiment
        ).astype(int)
        
        return features_df

    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        companies_df, financial_df, news_df = self.load_data()
        
        # Clean and process data
        financial_df = self.clean_financial_data(financial_df)
        news_stats = self.process_news_data(news_df)
        
        # Create features
        features_df = self.create_features(financial_df, news_stats)
        
        # Prepare training data
        training_data = self.prepare_training_data(features_df)
        
        # Save processed data
        output_file = os.path.join(self.processed_data_dir, 'processed_data.csv')
        training_data.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        return training_data

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_preprocessing() 