import logging
from src.preprocessing.financial_timeseries import FinancialTimeseriesPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('financial_preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting financial data preprocessing pipeline")
        
        # Initialize and run the preprocessor
        preprocessor = FinancialTimeseriesPreprocessor()
        symbols_data, clustering_data = preprocessor.run_preprocessing()
        
        # Log results
        if symbols_data:
            processed_count = sum(1 for ts, ratios in symbols_data.values() if ts is not None and ratios is not None)
            logger.info(f"Successfully processed time series data for {processed_count} companies")
        
        if clustering_data is not None:
            logger.info(f"Generated clustering data with {len(clustering_data)} records")
        
        logger.info("Financial data preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()