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

    def click_year_tab(self):
        """Click the 'Theo năm' tab to show yearly data"""
        try:
            # Find the year tab using its ID and class
            year_tab = self.wait.until(
                EC.element_to_be_clickable((By.ID, "idTabTaiChinhNam"))
            )
            year_tab.click()
            time.sleep(3)  # Wait for the page to load
            return True
        except Exception as e:
            logger.error(f"Error clicking year tab: {str(e)}")
            return False

    def extract_year_from_data(self, financial_data):
        """Extract unique years from financial data"""
        years = set()
        for data in financial_data:
            # Look for year patterns in the metric name
            metric = data['metric']
            if 'năm' in metric.lower():
                # Extract year from metric name (e.g., "Doanh thu năm 2023")
                import re
                year_match = re.search(r'năm\s+(\d{4})', metric.lower())
                if year_match:
                    years.add(int(year_match.group(1)))
        return sorted(years, reverse=True)

    def crawl_financial_data(self, symbol, exchange):
        """Crawl financial data for a company for the last 3 years"""
        logger.info(f"Crawling financial data for {symbol} ({exchange}) for the last 3 years...")
        
        # Construct URL based on exchange and symbol
        url = f"{self.base_url}/du-lieu/{exchange.lower()}/{symbol.lower()}.chn"
        logger.info(f"Accessing URL: {url}")
        
        try:
            # Get the page
            page_source = self.get_page(url)
            if not page_source:
                logger.error(f"Failed to get page for {symbol}")
                return
                
            # Click the year tab
            if not self.click_year_tab():
                logger.error(f"Failed to click year tab for {symbol}")
                return
                
            # Get the updated page source after clicking the tab
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract financial data
            financial_data = self.extract_financial_data(soup, symbol)
            
            if financial_data:
                df = pd.DataFrame(financial_data)
                output_file = os.path.join(self.data_dir, f'financial_data_{symbol}.csv')
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {len(financial_data)} financial records for {symbol} to {output_file}")
                
                # Log the years collected
                years = sorted(set(df['year']), reverse=True)
                logger.info(f"Collected data for years: {years}")
            else:
                logger.warning(f"No financial data found for {symbol}")
                
        except Exception as e:
            logger.error(f"Error crawling {symbol}: {str(e)}")

    def extract_financial_data(self, soup, symbol):
        """Extract financial data from the yearly table"""
        financial_data = []
        
        # Find the main table containing financial data
        table = soup.find('table', {'width': '100%', 'border': '0', 'cellspacing': '0', 'cellpadding': '0'})
        if not table:
            logger.warning("Could not find financial data table")
            return financial_data
            
        # Extract years from the header
        header_row = table.find('tr')
        if not header_row:
            logger.warning("Could not find header row")
            return financial_data
            
        years = []
        for th in header_row.find_all('th')[1:]:  # Skip the first column (metric names)
            year_text = th.get_text(strip=True)
            if 'Năm' in year_text:
                year = year_text.split('Năm')[1].strip().split()[0]
                years.append(year)
        
        # Extract data rows
        for row in table.find_all('tr')[1:]:  # Skip header row
            try:
                # Get metric name from first column
                metric_cell = row.find('td', {'class': 'col1'})
                if not metric_cell:
                    continue
                    
                metric_name = metric_cell.get_text(strip=True)
                
                # Get values for each year
                value_cells = row.find_all('td', {'style': 'text-align: right'})
                for i, cell in enumerate(value_cells):
                    if i >= len(years):
                        break
                        
                    value = cell.get_text(strip=True)
                    if value and value != '-':
                        data = {
                            'symbol': symbol,
                            'metric': metric_name,
                            'year': years[i],
                            'value': self.clean_number(value),
                            'crawled_date': datetime.now().strftime('%Y-%m-%d')
                        }
                        financial_data.append(data)
                        logger.info(f"Found {metric_name} for year {years[i]}: {value}")
                        
            except Exception as e:
                logger.error(f"Error parsing row: {str(e)}")
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