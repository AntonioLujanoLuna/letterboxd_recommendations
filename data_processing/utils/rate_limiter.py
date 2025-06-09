import asyncio
import time
from datetime import datetime
from typing import Optional

class RateLimiter:
    """Rate limiter for web scraping"""
    
    def __init__(self, requests_per_second: float = 2.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()

# Updated fetch function with rate limiting and retry logic
async def fetch_with_retry(url: str, session, rate_limiter: RateLimiter, 
                          max_retries: int = 3, input_data: dict = None):
    """
    Fetch URL with rate limiting, retries, and error handling
    """
    from utils.logging_config import setup_logger
    logger = setup_logger('letterboxd.scraper')
    
    for attempt in range(max_retries):
        try:
            # Apply rate limiting
            await rate_limiter.acquire()
            
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.read(), input_data
                elif response.status == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for {url}, attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Error fetching {url}, attempt {attempt + 1}: {str(e)}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None, input_data

# Updated letterboxd fetching with validation
async def fetch_letterboxd_with_validation(url, session, rate_limiter, input_data={}):
    """Fetch and validate Letterboxd movie data"""
    from utils.validation import DataValidator
    from utils.logging_config import setup_logger
    
    logger = setup_logger('letterboxd.scraper')
    
    response_data = await fetch_with_retry(url, session, rate_limiter, input_data=input_data)
    
    if response_data[0] is None:
        return None
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response_data[0], "lxml")
        
        movie_header = soup.find('section', attrs={'id': 'featured-film-header'})
        
        # Extract data with defaults
        movie_title = ''
        year = None
        imdb_id = ''
        tmdb_id = ''
        imdb_link = ''
        tmdb_link = ''
        
        if movie_header:
            try:
                title_elem = movie_header.find('h1')
                movie_title = title_elem.text.strip() if title_elem else ''
            except AttributeError:
                pass
            
            try:
                year_elem = movie_header.find('small', attrs={'class': 'number'})
                if year_elem:
                    year_link = year_elem.find('a')
                    if year_link:
                        year = int(year_link.text)
            except (AttributeError, ValueError):
                pass
        
        # Extract external IDs
        try:
            imdb_elem = soup.find("a", attrs={"data-track-action": "IMDb"})
            if imdb_elem:
                imdb_link = imdb_elem['href']
                imdb_id = imdb_link.split('/title')[1].strip('/').split('/')[0]
        except (KeyError, IndexError, AttributeError):
            pass
        
        try:
            tmdb_elem = soup.find("a", attrs={"data-track-action": "TMDb"})
            if tmdb_elem:
                tmdb_link = tmdb_elem['href']
                tmdb_id = tmdb_link.split('/movie')[1].strip('/').split('/')[0]
        except (KeyError, IndexError, AttributeError):
            pass
        
        # Create movie object
        movie_object = {
            "movie_id": input_data["movie_id"],
            "movie_title": movie_title,
            "year_released": year,
            "imdb_link": imdb_link,
            "tmdb_link": tmdb_link,
            "imdb_id": imdb_id,
            "tmdb_id": tmdb_id,
            "last_updated": datetime.now()
        }
        
        # Validate the data
        validated_movie, errors = DataValidator.validate_movie_data(movie_object)
        
        if errors:
            logger.warning(f"Validation errors for movie {input_data['movie_id']}: {errors}")
        
        from pymongo import UpdateOne
        return UpdateOne(
            {"movie_id": input_data["movie_id"]},
            {"$set": validated_movie},
            upsert=True
        )
        
    except Exception as e:
        logger.error(f"Error parsing movie data for {input_data['movie_id']}: {str(e)}")
        return None