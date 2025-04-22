import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('financial_preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

class FinancialPreprocessor:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.unscaled_dir = os.path.join(self.data_dir, 'unscaled')
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.unscaled_dir, exist_ok=True)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
        # Define metric mappings for easier access
        self.metric_mappings = {
            'Doanh thu': 'revenue',
            'Doanh thu bán hàng': 'revenue',
            'Doanh thu thuần': 'revenue',
            'Doanh thu bán hàng và CCDV': 'revenue',
            'Doanh thu thuần HĐKD bảo hiểm': 'revenue',
            'Doanh thu thuần hoạt động kinh doanh bảo hiểm': 'revenue',
            'Doanh thu thuần hoạt động kinh doanh': 'revenue',
            'Doanh thu thuần từ hoạt động kinh doanh': 'revenue',
            'Doanh thu thuần từ hoạt động kinh doanh bảo hiểm': 'revenue',
            'Giá vốn hàng bán': 'cost_of_goods_sold',
            'Lợi nhuận gộp': 'gross_profit',
            'Lợi nhuận gộp về BH và CCDV': 'gross_profit',
            'Lợi nhuận tài chính': 'financial_profit',
            'Lợi nhuận khác': 'other_profit',
            'Tổng lợi nhuận trước thuế': 'profit_before_tax',
            'Lợi nhuận sau thuế': 'net_income',
            'Lợi nhuận sau thuế của công ty mẹ': 'parent_company_profit',
            'Tổng tài sản lưu động ngắn hạn': 'current_assets',
            'Tổng tài sản': 'total_assets',
            'Nợ ngắn hạn': 'current_liabilities',
            'Tổng nợ': 'total_debt',
            'Vốn chủ sở hữu': 'equity'
        }

    def clean_value(self, value):
        """Clean and convert value to numeric format"""
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return value
        try:
            # Remove any non-numeric characters except decimal points and negative signs
            cleaned = ''.join(c for c in str(value) if c.isdigit() or c in '.-')
            return float(cleaned) if cleaned else 0
        except (ValueError, TypeError):
            return 0

    def get_financial_files(self):
        """Get all financial data files"""
        financial_files = []
        for file in os.listdir(self.raw_dir):
            if file.startswith('financial_data_') and file.endswith('.csv'):
                financial_files.append(file)
        return financial_files

    def load_financial_data(self, symbol):
        """Load financial data for a specific symbol"""
        file_path = os.path.join(self.raw_dir, f'financial_data_{symbol}.csv')
        if not os.path.exists(file_path):
            logger.warning(f"No financial data found for {symbol}")
            return None
            
        try:
            df = pd.read_csv(file_path)
            # Clean the value column
            df['value'] = df['value'].apply(self.clean_value)
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None

    def map_metric(self, metric):
        """Map a metric to its standardized name based on keyword matching"""
        metric = str(metric).lower()
        
        # Revenue metrics
        if 'doanh thu' in metric:
            return 'revenue'
            
        # Cost metrics
        if 'giá vốn' in metric:
            return 'cost_of_goods_sold'
            
        # Profit metrics
        if 'lợi nhuận gộp' in metric:
            return 'gross_profit'
        if 'lợi nhuận tài chính' in metric:
            return 'financial_profit'
        if 'lợi nhuận khác' in metric:
            return 'other_profit'
        if 'lợi nhuận trước thuế' in metric:
            return 'profit_before_tax'
        if 'lợi nhuận sau thuế' in metric:
            return 'net_income'
            
        # Asset metrics
        if 'tài sản lưu động' in metric:
            return 'current_assets'
        if 'tổng tài sản' in metric:
            return 'total_assets'
            
        # Liability metrics
        if 'nợ ngắn hạn' in metric:
            return 'current_liabilities'
        if 'tổng nợ' in metric:
            return 'total_debt'
            
        # Equity metrics
        if 'vốn chủ sở hữu' in metric:
            return 'equity'
            
        return None

    def transform_to_time_series(self, df):
        """Transform financial data into time series format"""
        try:
            # Map metrics to standardized names
            df['metric'] = df['metric'].apply(self.map_metric)
            
            # Drop rows where metric mapping failed
            df = df.dropna(subset=['metric'])
            
            if df.empty:
                logger.error("No valid data after metric mapping")
                return None
            
            # Convert year to integer
            df['year'] = df['year'].astype(int)
            
            # Remove duplicate entries for the same metric and year
            df = df.drop_duplicates(subset=['metric', 'year'], keep='first')
            
            # Pivot the data to get time series format
            ts_df = df.pivot(index='year', columns='metric', values='value')
            
            # Sort by year
            ts_df = ts_df.sort_index()
            
            return ts_df
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def calculate_ratios(self, ts_data):
        """Calculate financial ratios"""
        ratios = pd.DataFrame(index=ts_data.index)
        print("Calculating financial ratios...")
        
        # ROA (Return on Assets) = Net Income / Total Assets
        if 'net_income' in ts_data.columns and 'total_assets' in ts_data.columns:
            ratios['roa'] = ts_data['net_income'] / ts_data['total_assets']
        
        # ROE (Return on Equity) = Net Income / Equity
        if 'net_income' in ts_data.columns and 'equity' in ts_data.columns:
            ratios['roe'] = ts_data['net_income'] / ts_data['equity']
        
        # Debt Ratio = Total Debt / Total Assets
        if 'total_debt' in ts_data.columns and 'total_assets' in ts_data.columns:
            ratios['debt_ratio'] = ts_data['total_debt'] / ts_data['total_assets']
        
        # Current Ratio = Current Assets / Current Liabilities
        if 'current_assets' in ts_data.columns and 'current_liabilities' in ts_data.columns:
            ratios['current_ratio'] = ts_data['current_assets'] / ts_data['current_liabilities']
        
        # Gross Margin = Gross Profit / Revenue
        if 'gross_profit' in ts_data.columns and 'revenue' in ts_data.columns:
            ratios['gross_margin'] = ts_data['gross_profit'] / ts_data['revenue']
        
        # Net Profit Margin = Net Income / Revenue
        if 'net_income' in ts_data.columns and 'revenue' in ts_data.columns:
            ratios['net_profit_margin'] = ts_data['net_income'] / ts_data['revenue']
        
        # Revenue Growth (year-over-year)
        if 'revenue' in ts_data.columns:
            ratios['revenue_growth'] = ts_data['revenue'].pct_change()
        
        # Replace inf and -inf with NaN, then fill NaN with 0
        ratios = ratios.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print("Ratios calculated:")
        print(ratios)
        return ratios

    def prepare_clustering_data(self, ts_data, ratios):
        """Prepare data for clustering"""
        try:
            # Combine original metrics and ratios
            combined_data = pd.concat([ts_data, ratios])
            
            # Transpose to have years as rows and metrics as columns
            clustering_data = combined_data.T
            
            # Fill missing values with 0
            clustering_data = clustering_data.fillna(0)
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clustering_data)
            scaled_df = pd.DataFrame(scaled_data, index=clustering_data.index, columns=clustering_data.columns)
            
            return scaled_df
        except Exception as e:
            logger.error(f"Error preparing clustering data: {str(e)}")
            return None

    def process_company(self, symbol):
        """Process financial data for a single company"""
        try:
            logger.info(f"Processing company {symbol}")
            
            # Load raw data
            raw_data = self.load_financial_data(symbol)
            if raw_data is None:
                logger.error(f"Failed to load raw data for {symbol}")
                return None
            
            # Transform to time series
            ts_data = self.transform_to_time_series(raw_data)
            if ts_data is None:
                logger.error(f"Failed to transform time series data for {symbol}")
                return None
            
            # Calculate ratios
            ratios = self.calculate_ratios(ts_data)
            if ratios is None:
                logger.error(f"Failed to calculate ratios for {symbol}")
                return None
            
            # Save unscaled data
            if not self.save_unscaled_data(ts_data, ratios, symbol):
                logger.error(f"Failed to save unscaled data for {symbol}")
                return None
            
            # Prepare clustering data (scaled)
            clustering_data = self.prepare_clustering_data(ts_data, ratios)
            if clustering_data is None:
                logger.error(f"Failed to prepare clustering data for {symbol}")
                return None
            
            # Save scaled data
            output_file = os.path.join(self.processed_dir, f'processed_{symbol}.csv')
            clustering_data.to_csv(output_file)
            
            logger.info(f"Successfully processed data for {symbol}")
            return clustering_data
            
        except Exception as e:
            logger.error(f"Error processing company {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def process_all_companies(self):
        """Process financial data for all companies"""
        try:
            # Get all raw data files
            raw_files = [f for f in os.listdir(self.raw_dir) if f.startswith('financial_data_') and f.endswith('.csv')]
            symbols = [f.split('_')[2].split('.')[0] for f in raw_files]
            
            logger.info(f"Found {len(symbols)} companies to process")
            
            # Process each company
            success_count = 0
            for symbol in symbols:
                if self.process_company(symbol) is not None:
                    success_count += 1
            
            logger.info(f"Successfully processed {success_count} out of {len(symbols)} companies")
            
        except Exception as e:
            logger.error(f"Error processing all companies: {str(e)}")
            logger.error(traceback.format_exc())

    def save_unscaled_data(self, ts_data, ratios, symbol):
        """Save unscaled financial data and ratios in the same format as processed data"""
        try:
            # Combine original metrics and ratios
            combined_data = pd.concat([ts_data, ratios], axis=1)
            
            # Transpose to match processed data format (metrics as rows, years as columns)
            transposed_data = combined_data.T
            
            # Fill missing values with 0
            transposed_data = transposed_data.fillna(0)
            
            # Save to CSV in the unscaled directory
            output_file = os.path.join(self.unscaled_dir, f'{symbol}.csv')
            transposed_data.to_csv(output_file)
            logger.info(f"Saved unscaled data for {symbol} to {self.unscaled_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving unscaled data for {symbol}: {str(e)}")
            return False

if __name__ == "__main__":
    try:
        preprocessor = FinancialPreprocessor()
        preprocessor.process_all_companies()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        traceback.print_exc()