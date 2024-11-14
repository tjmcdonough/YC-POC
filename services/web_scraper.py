import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, List, Set
import re
from urllib.parse import urlparse, urljoin
import logging
from datetime import datetime
from queue import Queue
import time

def crawl_website(start_url: str, vector_store=None, llm_service=None) -> List[Dict]:
    scraper = WebScraperService()
    return scraper.crawl_website(start_url, vector_store, llm_service)


class WebScraperService:

    def __init__(self, max_pages: int = 100, rate_limit: float = 1.0):
        self.headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.max_pages = max_pages
        self.rate_limit = rate_limit
        self.visited_urls: Set[str] = set()
        self.domain = None

    def _validate_url(self, url: str) -> bool:
        """Validate if the URL is properly formatted"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _clean_text(self, text: str) -> str:
        """Clean and normalize scraped text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from the webpage"""
        metadata = {
            'url': url,
            'title': '',
            'description': '',
            'scraped_at': datetime.now().isoformat()
        }

        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.text.strip()

        # Extract meta description
        description_tag = soup.find('meta', attrs={'name': 'description'}) or \
                         soup.find('meta', attrs={'property': 'og:description'})
        if description_tag:
            metadata['description'] = description_tag.get('content',
                                                          '').strip()

        return metadata

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Extract all valid links from the page"""
        links = set()
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            absolute_url = urljoin(base_url, href)

            # Skip if not valid URL
            if not self._validate_url(absolute_url):
                continue

            # Skip if different domain
            if urlparse(absolute_url).netloc != self.domain:
                continue

            # Skip common non-content URLs
            if any(pattern in absolute_url.lower() for pattern in [
                    '.pdf', '.jpg', '.png', '.gif', '.css', '.js', '#',
                    'mailto:', 'tel:', 'javascript:', '/tag/', '/category/',
                    '/author/', '/search'
            ]):
                continue

            links.add(absolute_url)
        return links

    def scrape_url(self, url: str) -> Optional[Dict]:
        """
        Scrape content from a given URL
        Returns a dictionary containing the scraped text and metadata
        """
        if not self._validate_url(url):
            raise ValueError(f"Invalid URL format: {url}")

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer',
                                 'iframe']):
                element.decompose()

            # Extract main content
            content = []

            # Try to find main content area
            main_content = soup.find('main') or soup.find(
                'article') or soup.find(
                    'div', class_=re.compile(r'content|main|article'))

            if main_content:
                paragraphs = main_content.find_all(
                    ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            else:
                paragraphs = soup.find_all(
                    ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

            for p in paragraphs:
                text = p.get_text()
                if len(text.split()) > 3:  # Only include if more than 3 words
                    content.append(self._clean_text(text))

            if not content:
                return None

            # Extract links for crawling
            links = self._extract_links(soup, url)

            return {
                'text': '\n\n'.join(content),
                'metadata': self._extract_metadata(soup, url),
                'links': links
            }

        except requests.RequestException as e:
            logging.error(f"Error scraping URL {url}: {str(e)}")
            return None

    def crawl_website(self, start_url: str, vector_store=None, llm_service=None) -> List[Dict]:
        """
        Recursively crawl a website starting from the given URL
        Adds each page to vector store as it's crawled
        Returns a list of scraped content from all crawled pages
        """
        if not self._validate_url(start_url):
            raise ValueError(f"Invalid start URL: {start_url}")

        # Set domain for the crawl
        self.domain = urlparse(start_url).netloc

        # Initialize crawling queue and results
        url_queue = Queue()
        url_queue.put(start_url)
        results = []
        pages_crawled = 0

        while not url_queue.empty() and pages_crawled < self.max_pages:
            current_url = url_queue.get()

            # Skip if already visited
            if current_url in self.visited_urls:
                continue

            # Mark as visited
            self.visited_urls.add(current_url)

            # Scrape the page
            logging.info(f"Crawling: {current_url}")
            result = self.scrape_url(current_url)

            if result:
                # Add to vector store immediately if provided
                if vector_store is not None and result['text']:
                    try:
                        vector_store.add_documents(
                            text=result['text'],
                            metadata={
                                "filename": result['metadata']['url'],
                                "file_type": "web",
                                "title": result['metadata']['title'],
                                "description": result['metadata']['description'],
                                "created_at": time.time()
                            }
                        )
                        logging.info(f"Added to vector store: {result['metadata']['url']}")
                    except Exception as e:
                        logging.error(f"Error adding to vector store: {str(e)}")

                results.append(result)
                pages_crawled += 1

                # Add new links to queue
                for link in result['links']:
                    if link not in self.visited_urls:
                        url_queue.put(link)

            # Respect rate limiting
            time.sleep(self.rate_limit)

            logging.info(f"Pages crawled: {pages_crawled}")

        return results

    def scrape_multiple_urls(self, urls: List[str]) -> List[Dict]:
        """
        Scrape content from multiple URLs without crawling
        Returns a list of successful scrapes
        """
        results = []
        for url in urls:
            try:
                result = self.scrape_url(url)
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(f"Error processing URL {url}: {str(e)}")
                continue
        return results


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize scraper
    scraper = WebScraperService(
        max_pages=100,  # Maximum pages to crawl
        rate_limit=1.0  # Seconds between requests
    )

    # Start crawling
    results = scraper.crawl_website("https://example.com")

    # Process results
    print(f"Successfully crawled {len(results)} pages")
