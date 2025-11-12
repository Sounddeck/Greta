"""
Playwright-based OSINT Tools for CPAS5
Advanced web scraping for JavaScript-rendered pages
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)


class OSINTPlaywrightTools:
    """
    Playwright-based OSINT tools for scraping JavaScript-rendered pages
    """
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
        
        # Metrics
        self.metrics = {
            "pages_scraped": 0,
            "screenshots_taken": 0,
            "errors": 0,
            "total_time": 0.0
        }
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close_browser()
    
    async def start_browser(self):
        """Start Playwright browser"""
        if not self.playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            logger.info("Playwright browser started")
    
    async def close_browser(self):
        """Close Playwright browser"""
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
            logger.info("Playwright browser closed")
    
    async def scrape_page(
        self,
        url: str,
        wait_for: Optional[str] = None,
        execute_js: Optional[str] = None,
        screenshot: bool = False
    ) -> Dict[str, Any]:
        """
        Scrape a webpage using Playwright
        
        Args:
            url: URL to scrape
            wait_for: CSS selector to wait for before scraping
            execute_js: JavaScript code to execute on the page
            screenshot: Whether to take a screenshot
            
        Returns:
            Dict with scraped data
        """
        start_time = datetime.utcnow()
        
        try:
            # Ensure browser is started
            if not self.browser:
                await self.start_browser()
            
            # Create new page
            page = await self.browser.new_page()
            
            try:
                # Navigate to URL
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait for specific element if specified
                if wait_for:
                    try:
                        await page.wait_for_selector(wait_for, timeout=10000)
                    except PlaywrightTimeoutError:
                        logger.warning(f"Timeout waiting for selector: {wait_for}")
                
                # Execute custom JavaScript if provided
                js_result = None
                if execute_js:
                    js_result = await page.evaluate(execute_js)
                
                # Extract basic data
                title = await page.title()
                content = await page.content()
                url_final = page.url  # Final URL after redirects
                
                # Extract emails
                emails = await page.evaluate("""
                    () => {
                        const text = document.body.innerText;
                        const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
                        return [...new Set(text.match(emailRegex) || [])];
                    }
                """)
                
                # Extract phone numbers
                phones = await page.evaluate("""
                    () => {
                        const text = document.body.innerText;
                        const phoneRegex = /(\\+?\\d{1,3}[-\\.\\s]?)?\\(?\\d{3}\\)?[-\\.\\s]?\\d{3}[-\\.\\s]?\\d{4}/g;
                        return [...new Set(text.match(phoneRegex) || [])];
                    }
                """)
                
                # Extract links
                links = await page.evaluate("""
                    () => Array.from(document.querySelectorAll('a'))
                        .map(a => a.href)
                        .filter(href => href && href.startsWith('http'))
                """)
                
                # Extract social media links
                social_links = await page.evaluate("""
                    () => {
                        const links = Array.from(document.querySelectorAll('a')).map(a => a.href);
                        return {
                            linkedin: links.filter(l => l.includes('linkedin.com')),
                            twitter: links.filter(l => l.includes('twitter.com') || l.includes('x.com')),
                            facebook: links.filter(l => l.includes('facebook.com')),
                            instagram: links.filter(l => l.includes('instagram.com')),
                            github: links.filter(l => l.includes('github.com'))
                        };
                    }
                """)
                
                # Extract meta tags
                meta_tags = await page.evaluate("""
                    () => {
                        const metas = {};
                        document.querySelectorAll('meta').forEach(meta => {
                            const name = meta.getAttribute('name') || meta.getAttribute('property');
                            const content = meta.getAttribute('content');
                            if (name && content) {
                                metas[name] = content;
                            }
                        });
                        return metas;
                    }
                """)
                
                # Take screenshot if requested
                screenshot_data = None
                if screenshot:
                    screenshot_bytes = await page.screenshot(full_page=True)
                    screenshot_data = screenshot_bytes.hex()  # Convert to hex string
                    self.metrics["screenshots_taken"] += 1
                
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                self.metrics["pages_scraped"] += 1
                self.metrics["total_time"] += execution_time
                
                return {
                    "success": True,
                    "url": url,
                    "url_final": url_final,
                    "title": title,
                    "emails": emails,
                    "phone_numbers": phones,
                    "links": list(set(links)),
                    "social_links": social_links,
                    "meta_tags": meta_tags,
                    "content_length": len(content),
                    "js_result": js_result,
                    "screenshot": screenshot_data,
                    "execution_time": execution_time
                }
                
            finally:
                await page.close()
        
        except Exception as e:
            logger.error(f"Playwright scraping failed for {url}: {e}")
            self.metrics["errors"] += 1
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    async def scrape_multiple_pages(
        self,
        urls: List[str],
        wait_for: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple pages concurrently
        
        Args:
            urls: List of URLs to scrape
            wait_for: CSS selector to wait for
            
        Returns:
            List of scraping results
        """
        tasks = [self.scrape_page(url, wait_for=wait_for) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error dicts
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "url": urls[i],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def extract_structured_data(
        self,
        url: str,
        selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract structured data using CSS selectors
        
        Args:
            url: URL to scrape
            selectors: Dict mapping field names to CSS selectors
            
        Returns:
            Dict with extracted data
        """
        try:
            if not self.browser:
                await self.start_browser()
            
            page = await self.browser.new_page()
            
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                extracted_data = {}
                for field_name, selector in selectors.items():
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            extracted_data[field_name] = await element.inner_text()
                        else:
                            extracted_data[field_name] = None
                    except Exception as e:
                        logger.warning(f"Failed to extract {field_name}: {e}")
                        extracted_data[field_name] = None
                
                return {
                    "success": True,
                    "url": url,
                    "data": extracted_data
                }
                
            finally:
                await page.close()
        
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    async def monitor_network_requests(
        self,
        url: str,
        filter_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Monitor network requests made by a page
        
        Args:
            url: URL to monitor
            filter_pattern: Optional pattern to filter requests (e.g., "api")
            
        Returns:
            Dict with network request data
        """
        try:
            if not self.browser:
                await self.start_browser()
            
            page = await self.browser.new_page()
            requests = []
            
            # Set up request listener
            async def handle_request(request):
                if not filter_pattern or filter_pattern in request.url:
                    requests.append({
                        "url": request.url,
                        "method": request.method,
                        "resource_type": request.resource_type,
                        "headers": request.headers
                    })
            
            page.on("request", handle_request)
            
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait a bit for any delayed requests
                await asyncio.sleep(2)
                
                return {
                    "success": True,
                    "url": url,
                    "requests": requests,
                    "total_requests": len(requests)
                }
                
            finally:
                await page.close()
        
        except Exception as e:
            logger.error(f"Network monitoring failed: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get scraping metrics"""
        return {
            **self.metrics,
            "avg_time_per_page": self.metrics["total_time"] / max(self.metrics["pages_scraped"], 1)
        }

