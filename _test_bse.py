import logging
from src.scraper.financial_scraper import FinancialScraper

logging.basicConfig(level=logging.INFO)

scraper = FinancialScraper(headless=True)
files = scraper._scrape_bse_india("Reliance Industries", "2024", "annual-report", 1)
print("Files downloaded:", files)
