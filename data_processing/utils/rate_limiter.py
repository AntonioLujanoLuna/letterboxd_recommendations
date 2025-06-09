import asyncio
import time
from datetime import datetime
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter for web scraping"""
    
    def __init__(self, requests_per_second: float = 2.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()

class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt using exponential backoff"""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)

async def fetch_with_retry(url: str, session, rate_limiter: RateLimiter, 
                          retry_config: Optional[RetryConfig] = None,
                          input_data: Optional[Dict] = None,
                          **kwargs) -> tuple[Optional[bytes], Optional[Dict]]:
    """
    Fetch URL with rate limiting, retries, and comprehensive error handling
    
    Returns:
        Tuple of (response_data, input_data) or (None, input_data) on failure
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    if input_data is None:
        input_data = {}
    
    for attempt in range(retry_config.max_retries):
        try:
            # Apply rate limiting
            await rate_limiter.acquire()
            
            # Make request with timeout
            timeout = kwargs.get('timeout', 30)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.read()
                    return data, input_data
                    
                elif response.status == 429:  # Rate limited
                    wait_time = retry_config.get_delay(attempt)
                    logger.warning(f"Rate limited (429), waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                    continue
                    
                elif response.status == 404:
                    logger.warning(f"Not found (404): {url}")
                    return None, input_data
                    
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    if response.status >= 500:  # Server error, retry
                        continue
                    else:  # Client error, don't retry
                        return None, input_data
                        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for {url}, attempt {attempt + 1}/{retry_config.max_retries}")
        except Exception as e:
            logger.error(f"Error fetching {url}, attempt {attempt + 1}: {str(e)}")
        
        # Wait before retry (except on last attempt)
        if attempt < retry_config.max_retries - 1:
            wait_time = retry_config.get_delay(attempt)
            await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to fetch {url} after {retry_config.max_retries} attempts")
    return None, input_data

class ConcurrencyManager:
    """Manages concurrent requests with proper resource limits"""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        self.active_requests += 1
        self.total_requests += 1
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.active_requests -= 1
        if exc_type is None:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.semaphore.release()
    
    def get_stats(self) -> Dict[str, int]:
        """Get current statistics"""
        return {
            'active': self.active_requests,
            'total': self.total_requests,
            'successful': self.successful_requests,
            'failed': self.failed_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1)
        }

async def process_urls_batch(urls_with_data: list, session, rate_limiter: RateLimiter,
                           concurrency_manager: ConcurrencyManager,
                           processor_func,
                           retry_config: Optional[RetryConfig] = None) -> list:
    """
    Process a batch of URLs with proper concurrency control
    
    Args:
        urls_with_data: List of (url, input_data) tuples
        session: aiohttp session
        rate_limiter: Rate limiter instance
        concurrency_manager: Concurrency manager
        processor_func: Function to process response data
        retry_config: Retry configuration
    
    Returns:
        List of processed results
    """
    async def process_single_url(url_data):
        url, input_data = url_data
        async with concurrency_manager:
            try:
                # Fetch data
                response_data, input_data = await fetch_with_retry(
                    url, session, rate_limiter, retry_config, input_data
                )
                
                # Process response if successful
                if response_data:
                    return await processor_func(response_data, input_data)
                else:
                    return None
                    
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                return None
    
    # Create tasks for all URLs
    tasks = [process_single_url(url_data) for url_data in urls_with_data]
    
    # Execute with proper error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and None results
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
        elif result is not None:
            processed_results.append(result)
    
    return processed_results