import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialCrawler:
    def __init__(self):
        self.base_url = "https://cafef.vn"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Referer': 'https://cafef.vn'
        }
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize Selenium WebDriver with webdriver-manager
        options = Options()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 20)  # Increased timeout to 20 seconds
        self.max_retries = 3  # Maximum number of retries for failed requests

    def get_page(self, url):
        """Fetch a page using Selenium with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                self.driver.get(url)
                # Wait for the page to load with increased timeout
                time.sleep(10)  # Increased wait time to 10 seconds
                return self.driver.page_source
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts: {str(e)}")
                    return None
                logger.warning(f"Attempt {attempt + 1} failed for {url}, retrying...")
                time.sleep(5)  # Wait before retrying

    def click_previous_button(self):
        """Click the previous button to get historical data"""
        try:
            # Find the previous button using the onclick attribute
            prev_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@onclick, 'ViewPage(1)')]"))
            )
            prev_button.click()
            time.sleep(5)  # Wait for the page to load
            return True
        except Exception as e:
            logger.error(f"Error clicking previous button: {str(e)}")
            return False

    def crawl_financial_data(self, symbol, exchange):
        """Crawl financial data for a company with pagination"""
        logger.info(f"Crawling financial data for {symbol} ({exchange})...")
        
        # Construct URL based on exchange and symbol
        url = f"{self.base_url}/du-lieu/{exchange.lower()}/{symbol.lower()}.chn"
        logger.info(f"Accessing URL: {url}")
        
        all_financial_data = []
        page_count = 0
        max_pages = 20  # Increased to get more historical data
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while page_count < max_pages and consecutive_failures < max_consecutive_failures:
            try:
                # Get the current page
                page_source = self.get_page(url)
                if not page_source:
                    consecutive_failures += 1
                    logger.warning(f"Failed to get page {page_count + 1} for {symbol}")
                    continue
                
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Extract financial data from tables
                financial_data = self.extract_financial_data(soup, symbol)
                if financial_data:
                    all_financial_data.extend(financial_data)
                    logger.info(f"Found {len(financial_data)} financial records on page {page_count + 1}")
                else:
                    logger.warning(f"No financial data found on page {page_count + 1}")
                
                # Try to click the previous button
                if not self.click_previous_button():
                    logger.info(f"No more previous pages available for {symbol}")
                    break
                
                page_count += 1
                consecutive_failures = 0  # Reset failure count on successful page
                logger.info(f"Processed page {page_count} for {symbol}")
                
                # Add delay between requests
                time.sleep(3)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Error processing page {page_count + 1} for {symbol}: {str(e)}")
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures for {symbol}, stopping")
                    break
                time.sleep(5)  # Wait longer before retrying after an error
        
        if all_financial_data:
            df = pd.DataFrame(all_financial_data)
            output_file = os.path.join(self.data_dir, f'financial_data_{symbol}.csv')
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(all_financial_data)} financial records for {symbol} to {output_file}")
        else:
            logger.warning(f"No financial data found for {symbol} after {page_count} pages")

    def extract_financial_data(self, soup, symbol):
        """Extract financial data from the page"""
        financial_data = []
        
        # Find all tables with financial data
        tables = soup.find_all('table')
        
        for table in tables:
            try:
                rows = table.find_all('tr')
                for row in rows:
                    try:
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            metric_name = cols[0].get_text(strip=True)
                            value = cols[1].get_text(strip=True)
                            
                            if metric_name and value and not value.startswith('Hãy đăng nhập'):
                                # Clean up the metric name
                                metric_name = metric_name.replace('\n', ' ').replace('\r', '')
                                
                                # Skip rows that don't contain financial data
                                if any(skip in metric_name.lower() for skip in ['trang', 'xem', 'đăng nhập']):
                                    continue
                                
                                data = {
                                    'symbol': symbol,
                                    'metric': metric_name,
                                    'value': self.clean_number(value),
                                    'crawled_date': datetime.now().strftime('%Y-%m-%d')
                                }
                                financial_data.append(data)
                                logger.info(f"Found metric: {metric_name} = {value}")
                    except Exception as e:
                        logger.error(f"Error parsing row for {symbol}: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Error parsing table for {symbol}: {str(e)}")
                continue
        
        return financial_data

    def clean_number(self, text):
        """Clean and format number strings"""
        try:
            # Remove any commas and whitespace
            cleaned = text.strip().replace(',', '')
            # Convert to float if possible
            return float(cleaned) if cleaned and cleaned != '-' else None
        except:
            return None

    def process_companies_csv(self, csv_file):
        """Process companies from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                self.crawl_financial_data(row['symbol'], row['exchange'])
                time.sleep(2)  # Add delay between requests
        finally:
            self.driver.quit()

if __name__ == "__main__":
    crawler = FinancialCrawler()
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw', 'all_companies.csv')
    crawler.process_companies_csv(csv_file)