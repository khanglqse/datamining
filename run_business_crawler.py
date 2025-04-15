from src.crawler.business_crawler import BusinessCrawler
import logging
import pandas as pd
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize the crawler
    crawler = BusinessCrawler()
    
    try:
        # Read company symbols from the all_companies.csv file
        companies_file = os.path.join('data', 'raw', 'all_companies.csv')
        if not os.path.exists(companies_file):
            logging.error("Companies file not found. Please run the main crawler first.")
            return
            
        companies_df = pd.read_csv(companies_file)
        symbols = companies_df['symbol'].tolist()
        
        # Crawl business data for all companies
        logging.info(f"Starting to crawl business data for {len(symbols)} companies...")
        crawler.crawl_all_business_data(symbols)
        
        logging.info("Completed crawling business data for all companies")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        # Make sure to close the browser
        crawler.close()

if __name__ == "__main__":
    main() 