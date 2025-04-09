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

class CafefCrawler:
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

    def get_all_companies(self):
        """Get list of all companies from the API endpoint"""
        logger.info("Getting list of all companies from API...")
        
        url = "https://cafef.vn/du-lieu/ajax/pagenew/databusiness/congtyniemyet.ashx?centerid=0&skip=0&take=2000&major=0"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('Success'):
                logger.error("API request failed")
                return None
                
            companies = []
            for company in data.get('Data', []):
                # Determine exchange based on TradeCenterId
                exchange = 'HOSE'  # Default to HOSE
                if company.get('TradeCenterId') == 2:
                    exchange = 'HNX'
                elif company.get('TradeCenterId') == 8:
                    exchange = 'UPCOM'
                
                company_data = {
                    'symbol': company.get('Symbol', ''),
                    'name': company.get('CompanyName', ''),
                    'exchange': exchange,
                    'category': company.get('CategoryName', ''),
                    'price': company.get('Price', 0),
                    'crawled_date': datetime.now().strftime('%Y-%m-%d')
                }
                companies.append(company_data)
                logger.info(f"Added company: {company_data['symbol']} - {company_data['name']} ({exchange})")
            
            df = pd.DataFrame(companies)
            output_file = os.path.join(self.data_dir, 'all_companies.csv')
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(companies)} companies to {output_file}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching companies from API: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing companies data: {str(e)}")
            return None


    def crawl_news(self, company_symbol):
        """Crawl news articles related to a specific company"""
        logger.info(f"Crawling news for {company_symbol}...")
        
        url = f"{self.base_url}/doanh-nghiep/{company_symbol}/tin-doanh-nghiep"
        self.driver.get(url)
        time.sleep(5)  # Wait for the page to load
        
        news_data = []
        try:
            # Wait for news items to load
            news_items = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.knswli"))
            )
            
            for item in news_items:
                try:
                    title_elem = item.find_element(By.CSS_SELECTOR, "a.knswli-title")
                    date_elem = item.find_element(By.CSS_SELECTOR, "span.knswli-time")
                    summary_elem = item.find_element(By.CSS_SELECTOR, "div.knswli-sapo")
                    
                    news = {
                        'symbol': company_symbol,
                        'title': title_elem.text.strip(),
                        'date': date_elem.text.strip(),
                        'summary': summary_elem.text.strip(),
                        'url': title_elem.get_attribute("href"),
                        'crawled_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    news_data.append(news)
                except Exception as e:
                    logger.error(f"Error parsing news item: {str(e)}")
                    continue
                    
        except TimeoutException:
            logger.error(f"Timeout waiting for news to load for {company_symbol}")
        except Exception as e:
            logger.error(f"Error getting news for {company_symbol}: {str(e)}")
        
        if news_data:
            df = pd.DataFrame(news_data)
            output_file = os.path.join(self.data_dir, f'news_{company_symbol}.csv')
            df.to_csv(output_file, index=False)
            logger.info(f"Saved news for {company_symbol} to {output_file}")
        return news_data

    def crawl_all_companies(self):
        """Crawl data for all companies"""
        try:
            logger.info("Starting to crawl data for all companies...")
            
            # Get list of all companies
            companies_df = self.get_all_companies()
            if companies_df is None or len(companies_df) == 0:
                logger.error("Failed to get companies list")
                return
            
            # Crawl data for each company
            for _, company in tqdm(companies_df.iterrows(), total=len(companies_df)):
                symbol = company['symbol']
                exchange = company['exchange']
                logger.info(f"Processing company: {symbol} ({exchange})")
                
                # Add delay between requests
                time.sleep(2)
                
                # Crawl financial statements
                self.crawl_financial_statements(symbol)
                
                # Add delay between requests
                time.sleep(2)
                
                # Crawl news
                self.crawl_news(symbol)
            
            logger.info("Completed crawling data for all companies")
            
        finally:
            # Always close the browser
            self.driver.quit()

    def crawl_financial_statements_from_csv(self, row):
        """Crawl financial statements for a company using CSV row data"""
        symbol = row['symbol']
        name = row['name']
        exchange = row['exchange']
        logger.info(f"Crawling financial data for {symbol}")
        
        # Construct URL based on exchange and symbol
        # If name is empty, use only the symbol
        if pd.isna(name) or not name.strip():
            url = f"{self.base_url}/du-lieu/{exchange.lower()}/{symbol.lower()}.chn"
        else:
            url = f"{self.base_url}/du-lieu/{exchange.lower()}/{symbol.lower()}-{self.slugify(name)}.chn"
        logger.info(f"Accessing URL: {url}")
        
        financial_data = []
        try:
            # Get the page using requests
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
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
                
            if financial_data:
                df = pd.DataFrame(financial_data)
                output_file = os.path.join(self.data_dir, f'financial_data_{symbol}.csv')
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {len(financial_data)} financial records for {symbol} to {output_file}")
            else:
                logger.warning(f"No financial data found for {symbol}")
                # Log the HTML for debugging
                logger.debug(f"Page HTML: {response.text[:1000]}...")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting {url}: {str(e)}")
        except Exception as e:
            logger.error(f"Error crawling {symbol}: {str(e)}")
        
        return financial_data

    def slugify(self, text):
        """Convert company name to URL slug format"""
        # Handle Vietnamese characters
        vietnamese_chars = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'đ': 'd',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y'
        }
        
        # Convert to lowercase and replace Vietnamese characters
        text = text.lower()
        for vi, en in vietnamese_chars.items():
            text = text.replace(vi, en)
        
        # Replace spaces with hyphens and remove special characters
        text = text.replace(' ', '-')
        text = ''.join(c for c in text if c.isalnum() or c == '-')
        
        # Remove multiple consecutive hyphens
        while '--' in text:
            text = text.replace('--', '-')
        
        return text.strip('-')

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
                self.crawl_financial_statements_from_csv(row)
                time.sleep(2)  # Add delay between requests
        finally:
            self.driver.quit()

if __name__ == "__main__":
    crawler = CafefCrawler()
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw', 'all_companies.csv')
    crawler.process_companies_csv(csv_file) 