import os
import sys
from crawler.financial_crawler import FinancialCrawler

def main():
    # Get the absolute path of the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add the project root to the Python path
    sys.path.append(project_root)
    
    # Initialize the crawler
    crawler = FinancialCrawler()
    
    # Path to the companies CSV file
    csv_file = os.path.join(project_root, 'data', 'raw', 'all_companies.csv')
    
    # Process all companies from the CSV file
    crawler.process_companies_csv(csv_file)

if __name__ == "__main__":
    main() 