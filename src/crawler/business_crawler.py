from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import logging
import os
from datetime import datetime

class BusinessCrawler:
    def __init__(self):
        self.base_url = "https://cafef.vn"
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize Selenium WebDriver
        options = Options()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 20)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_business_data(self, symbol):
        """Get business data for a specific company"""
        self.logger.info(f"Crawling business data for {symbol}...")
        
        url = f"{self.base_url}/doanh-nghiep/{symbol}/tin-doanh-nghiep"
        self.driver.get(url)
        time.sleep(5)  # Wait for the page to load
        
        business_data = []
        try:
            # Wait for business data section to load
            business_section = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.knswli"))
            )
            
            # Get business information
            business_info = {
                'symbol': symbol,
                'company_name': self._get_company_name(),
                'business_type': self._get_business_type(),
                'industry': self._get_industry(),
                'market_cap': self._get_market_cap(),
                'revenue': self._get_revenue(),
                'profit': self._get_profit(),
                'employees': self._get_employees(),
                'crawled_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            business_data.append(business_info)
            self.logger.info(f"Successfully crawled business data for {symbol}")
            
        except TimeoutException:
            self.logger.error(f"Timeout waiting for business data to load for {symbol}")
        except Exception as e:
            self.logger.error(f"Error getting business data for {symbol}: {str(e)}")
        
        if business_data:
            df = pd.DataFrame(business_data)
            output_file = os.path.join(self.data_dir, f'business_data_{symbol}.csv')
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved business data for {symbol} to {output_file}")
        
        return business_data

    def _get_company_name(self):
        """Extract company name from the page"""
        try:
            name_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1.title"))
            )
            return name_element.text.strip()
        except:
            return None

    def _get_business_type(self):
        """Extract business type from the page"""
        try:
            business_type_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.business-type"))
            )
            return business_type_element.text.strip()
        except:
            return None

    def _get_industry(self):
        """Extract industry from the page"""
        try:
            industry_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.industry"))
            )
            return industry_element.text.strip()
        except:
            return None

    def _get_market_cap(self):
        """Extract market capitalization from the page"""
        try:
            market_cap_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.market-cap"))
            )
            return self._clean_number(market_cap_element.text.strip())
        except:
            return None

    def _get_revenue(self):
        """Extract revenue from the page"""
        try:
            revenue_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.revenue"))
            )
            return self._clean_number(revenue_element.text.strip())
        except:
            return None

    def _get_profit(self):
        """Extract profit from the page"""
        try:
            profit_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.profit"))
            )
            return self._clean_number(profit_element.text.strip())
        except:
            return None

    def _get_employees(self):
        """Extract number of employees from the page"""
        try:
            employees_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.employees"))
            )
            return self._clean_number(employees_element.text.strip())
        except:
            return None

    def _clean_number(self, text):
        """Clean and format number strings"""
        try:
            # Remove any commas and whitespace
            cleaned = text.strip().replace(',', '')
            # Convert to float if possible
            return float(cleaned) if cleaned and cleaned != '-' else None
        except:
            return None

    def crawl_all_business_data(self, symbols):
        """Crawl business data for multiple companies"""
        for symbol in symbols:
            self.get_business_data(symbol)
            time.sleep(2)  # Add delay between requests

    def close(self):
        """Close the browser"""
        self.driver.quit() 