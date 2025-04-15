from src.crawler.cafef_crawler import CafefCrawler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize the crawler
    crawler = CafefCrawler()
    
    try:
        # First, get all companies
        logging.info("Starting to get all companies...")
        companies_df = crawler.get_all_companies()
        
        if companies_df is not None:
            logging.info(f"Successfully retrieved {len(companies_df)} companies")
            
            # Now crawl data for all companies
            logging.info("Starting to crawl data for all companies...")
            crawler.crawl_all_companies()
            
        else:
            logging.error("Failed to get companies list")
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        # Make sure to close the browser
        crawler.driver.quit()

if __name__ == "__main__":
    main() 