import requests
from bs4 import BeautifulSoup

def scrape_website(url: str):
    """Fetch and return website HTML content."""
    import requests
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.text

