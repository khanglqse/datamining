import os
import logging
from src.crawler.cafef_crawler import CafefCrawler
from src.preprocessing.preprocess import DataPreprocessor
from src.modeling.train import ModelTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the complete data mining pipeline"""
    logger.info("Starting data mining pipeline...")
    
    try:
        # Step 1: Data Collection
        logger.info("Step 1: Data Collection")
        crawler = CafefCrawler()
        crawler.run_full_crawl()
        
        # Step 2: Data Preprocessing
        logger.info("Step 2: Data Preprocessing")
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.run_preprocessing()
        
        # Step 3: Model Training
        logger.info("Step 3: Model Training")
        trainer = ModelTrainer()
        trainer.run_training()
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 