import unittest
from src.web_scraper import scrape_website
from unittest.mock import patch

class TestWebScraper(unittest.TestCase):
    
    @patch('src.web_scraper.requests.get')
    def test_scrape_website_phishing(self, mock_get):
        # Simulate a phishing site response
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = '<html><body><a href="http://phishingsite.com">Click here</a></body></html>'

        url = 'http://example-phishing.com'
        result = scrape_website(url)
        
        self.assertIn('phishing', result.lower(), "Website was not identified as phishing!")

    @patch('src.web_scraper.requests.get')
    def test_scrape_website_legitimate(self, mock_get):
        # Simulate a legitimate site response
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = '<html><body><a href="http://legitwebsite.com">Visit our site</a></body></html>'

        url = 'http://example-legitimate.com'
        result = scrape_website(url)
        
        self.assertNotIn('phishing', result.lower(), "Website was incorrectly identified as phishing!")

if __name__ == '__main__':
    unittest.main()