#!/usr/local/bin/python3.11

import datetime
from typing import Optional
from bs4 import BeautifulSoup
import asyncio
from aiohttp import ClientSession
import pymongo
import pandas as pd
import time
from tqdm import tqdm
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError
from pprint import pprint
from .db_connect import DatabaseConnection, connect_to_db, get_database
from config import Config   
from .utils.rate_limiter import RateLimiter, fetch_with_retry, ConcurrencyManager, process_urls_batch, RetryConfig

logger = None  # Will be set by setup_logger if available

try:
    from .utils.logging_config import setup_logger
    logger = setup_logger('letterboxd.movies')
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

async def get_movies(movie_list, db_collection):  # Changed: removed mongo_db, renamed db_cursor to db_collection
    if not movie_list:
        return

    # Setup rate limiting and concurrency
    rate_limiter = RateLimiter(requests_per_second=1/Config.scraping.request_delay)
    concurrency_manager = ConcurrencyManager(max_concurrent=Config.scraping.concurrent_requests)
    retry_config = RetryConfig(max_retries=Config.scraping.max_retries)
    
    # Prepare URLs and data
    urls_with_data = [
        (f"https://letterboxd.com/film/{movie}/", {"movie_id": movie})
        for movie in movie_list
    ]
    
    async with ClientSession() as session:
        logger.info(f"Starting to scrape {len(movie_list)} movies")
        
        # Process URLs in batch
        update_operations = await process_urls_batch(
            urls_with_data,
            session,
            rate_limiter,
            concurrency_manager,
            process_movie_response,
            retry_config
        )
        
        # Log statistics
        stats = concurrency_manager.get_stats()
        logger.info(f"Completed batch: {stats}")

    # Bulk write to database
    if update_operations:
        try:
            db_collection.bulk_write(update_operations, ordered=False)  # Now uses the parameter
            logger.info(f"Successfully updated {len(update_operations)} movies in database")
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")

async def get_movie_posters(movie_list, db_collection):  # Changed parameters
    if not movie_list:
        return

    rate_limiter = RateLimiter(requests_per_second=1/Config.scraping.request_delay)
    concurrency_manager = ConcurrencyManager(max_concurrent=Config.scraping.concurrent_requests)
    retry_config = RetryConfig(max_retries=Config.scraping.max_retries)
    
    urls_with_data = [
        (f"https://letterboxd.com/ajax/poster/film/{movie}/hero/230x345", {"movie_id": movie})
        for movie in movie_list
    ]
    
    async with ClientSession() as session:
        logger.info(f"Starting to scrape posters for {len(movie_list)} movies")
        
        update_operations = await process_urls_batch(
            urls_with_data,
            session,
            rate_limiter,
            concurrency_manager,
            process_poster_response,
            retry_config
        )

    if update_operations:
        try:
            db_collection.bulk_write(update_operations, ordered=False)  # Now uses the parameter
            logger.info(f"Successfully updated {len(update_operations)} movie posters")
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")

async def get_rich_data(movie_list, db_collection, mongo_db, tmdb_key):
    """
    Fetch additional movie data from TMDB API
    
    Args:
        movie_list: List of movie IDs (strings) or movie documents (dicts)
        db_collection: MongoDB movies collection
        mongo_db: MongoDB database instance
        tmdb_key: TMDB API key
    """
    if not movie_list or not tmdb_key:
        logger.warning("No movies to process or missing TMDB key")
        return
        
    base_url = "https://api.themoviedb.org/3/movie/{}?api_key={}"
    
    # Process movie list to extract valid TMDB IDs
    movies_with_tmdb = []
    
    for item in movie_list:
        movie_doc = None
        
        if isinstance(item, dict):
            # It's already a movie document
            movie_doc = item
        elif isinstance(item, str):
            # It's a movie ID, fetch from database
            movie_doc = db_collection.find_one({"movie_id": item})
        else:
            logger.warning(f"Invalid movie item type: {type(item)}")
            continue
        
        # Validate TMDB ID
        if movie_doc and movie_doc.get('tmdb_id'):
            tmdb_id = str(movie_doc['tmdb_id']).strip()
            # Validate it's not empty and looks like a valid ID
            if tmdb_id and tmdb_id.isdigit():
                movies_with_tmdb.append(movie_doc)
            else:
                logger.debug(f"Invalid TMDB ID for movie {movie_doc.get('movie_id')}: '{tmdb_id}'")
    
    if not movies_with_tmdb:
        logger.info("No movies with valid TMDB IDs found")
        return
    
    logger.info(f"Fetching TMDB data for {len(movies_with_tmdb)} movies")
    
    # Setup rate limiting for TMDB API (they have strict limits)
    rate_limiter = RateLimiter(requests_per_second=4)  # TMDB allows ~4 requests/second
    retry_config = RetryConfig(max_retries=3, base_delay=1.0)
    
    async with ClientSession() as session:
        tasks = []
        
        for movie in movies_with_tmdb:
            try:
                url = base_url.format(movie["tmdb_id"], tmdb_key)
                task = asyncio.ensure_future(
                    fetch_with_retry(
                        url,
                        session,
                        rate_limiter,
                        retry_config,
                        input_data={"movie_id": movie["movie_id"], "movie_doc": movie}
                    )
                )
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating task for movie {movie.get('movie_id')}: {str(e)}")
                continue
        
        # Process responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        update_operations = []
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Task failed with exception: {response}")
                continue
                
            if response and response[0]:  # response is (data, input_data)
                try:
                    response_data, input_data = response
                    operation = await process_tmdb_response(response_data, input_data['movie_doc'])
                    if operation:
                        update_operations.append(operation)
                except Exception as e:
                    logger.error(f"Error processing TMDB response: {str(e)}")

    # Bulk write to database
    if update_operations:
        try:
            db_collection.bulk_write(update_operations, ordered=False)
            logger.info(f"Successfully updated {len(update_operations)} movies with TMDB data")
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")
            # Log some details about what failed
            if hasattr(bwe, 'details') and 'writeErrors' in bwe.details:
                for error in bwe.details['writeErrors'][:5]:  # Show first 5 errors
                    logger.error(f"Write error: {error}")

async def process_movie_response(response_data: bytes, input_data: dict) -> Optional[UpdateOne]:
    """Process movie response data and create database update operation"""
    try:
        soup = BeautifulSoup(response_data, "lxml")
        movie_header = soup.find('section', attrs={'id': 'featured-film-header'})

        # Extract data with better error handling
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

        # Extract external IDs with better error handling
        try:
            imdb_elem = soup.find("a", attrs={"data-track-action": "IMDb"})
            if imdb_elem and 'href' in imdb_elem.attrs:
                imdb_link = imdb_elem['href']
                imdb_id = imdb_link.split('/title')[1].strip('/').split('/')[0]
        except (KeyError, IndexError, AttributeError):
            pass

        try:
            tmdb_elem = soup.find("a", attrs={"data-track-action": "TMDb"})
            if tmdb_elem and 'href' in tmdb_elem.attrs:
                tmdb_link = tmdb_elem['href']
                tmdb_id = tmdb_link.split('/movie')[1].strip('/').split('/')[0]
        except (KeyError, IndexError, AttributeError):
            pass

        movie_object = {
            "movie_id": input_data["movie_id"],
            "movie_title": movie_title,
            "year_released": year,
            "imdb_link": imdb_link,
            "tmdb_link": tmdb_link,
            "imdb_id": imdb_id,
            "tmdb_id": tmdb_id,
            "last_updated": datetime.datetime.now()
        }

        # Validate data using config
        if not Config.validate_movie_title(movie_title):
            logger.warning(f"Invalid movie title for {input_data['movie_id']}")
            movie_object['movie_title'] = movie_title[:Config.validation.max_movie_title_length]
        
        if not Config.validate_year(year):
            logger.warning(f"Invalid year {year} for movie {input_data['movie_id']}")
            movie_object['year_released'] = None

        return UpdateOne(
            {"movie_id": input_data["movie_id"]},
            {"$set": movie_object},
            upsert=True
        )
        
    except Exception as e:
        logger.error(f"Error processing movie {input_data['movie_id']}: {str(e)}")
        return None

async def process_poster_response(response_data: bytes, input_data: dict) -> Optional[UpdateOne]:
    """Process poster response data"""
    try:
        soup = BeautifulSoup(response_data, "lxml")

        try:
            image_url = soup.find('div', attrs={'class': 'film-poster'}).find('img')['src'].split('?')[0]
            image_url = image_url.replace('https://a.ltrbxd.com/resized/', '').split('.jpg')[0]
            
            if 'https://s.ltrbxd.com/static/img/empty-poster' in image_url:
                image_url = ''
        except AttributeError:
            image_url = ''

        movie_object = {
            "movie_id": input_data["movie_id"],
            'last_updated': datetime.datetime.now()
        }

        if image_url:
            movie_object["image_url"] = image_url

        return UpdateOne(
            {"movie_id": input_data["movie_id"]},
            {"$set": movie_object},
            upsert=True
        )

    except Exception as e:
        logger.error(f"Error processing poster for {input_data['movie_id']}: {str(e)}")
        return None
    
async def process_tmdb_response(response_data: bytes, movie_doc: dict) -> Optional[UpdateOne]:
    """Process TMDB API response data"""
    try:
        import json
        response = json.loads(response_data.decode('utf-8'))
        
        # Start with existing movie data
        movie_object = {
            "movie_id": movie_doc["movie_id"],
            "tmdb_id": movie_doc["tmdb_id"]
        }
        
        # Extract object fields
        object_fields = ["genres", "production_countries", "spoken_languages"]
        for field_name in object_fields:
            try:
                if field_name in response and response[field_name]:
                    movie_object[field_name] = [x["name"] for x in response[field_name]]
                else:
                    movie_object[field_name] = []
            except (KeyError, TypeError) as e:
                logger.debug(f"Could not extract {field_name}: {e}")
                movie_object[field_name] = []
        
        # Extract simple fields
        simple_fields = ["popularity", "overview", "runtime", "vote_average", 
                        "vote_count", "release_date", "original_language"]
        for field_name in simple_fields:
            try:
                movie_object[field_name] = response.get(field_name)
            except Exception as e:
                logger.debug(f"Could not extract {field_name}: {e}")
                movie_object[field_name] = None
        
        movie_object['last_updated'] = datetime.datetime.now()
        
        return UpdateOne(
            {"movie_id": movie_doc["movie_id"]},
            {"$set": movie_object},
            upsert=True
        )

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response for movie {movie_doc.get('movie_id')}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing TMDB data for {movie_doc.get('movie_id')}: {str(e)}")
        return None

def main(data_type="letterboxd"):
    """Main function with improved error handling"""
    try:
        with get_database() as (db, tmdb_key):
            movies = db.movies
            
            # Find movies to scrape based on data type
            if data_type == "letterboxd":
                newly_added = [x['movie_id'] for x in movies.find({"tmdb_id": {"$exists": False}})]
                needs_update = [x['movie_id'] for x in movies.find({"tmdb_id": {"$exists": True}}).sort("last_updated", -1).limit(6000)]
                all_movies = needs_update + newly_added
            elif data_type == "poster":
                two_months_ago = datetime.datetime.now() - datetime.timedelta(days=60)
                all_movies = [x['movie_id'] for x in movies.find({"$or": [{"image_url": {"$exists": False}}, {"last_updated": {"$lte": two_months_ago}}]})]
            else:
                all_movies = [x['movie_id'] for x in movies.find({"genres": {"$exists": False}, "tmdb_id": {"$ne": ""}, "tmdb_id": {"$exists": True}})]
            
            # Process in chunks
            chunk_size = Config.scraping.chunk_size_movies
            chunks = [all_movies[i:i+chunk_size] for i in range(0, len(all_movies), chunk_size)]

            logger.info(f"Total Movies to Scrape: {len(all_movies)}")
            logger.info(f"Total Chunks: {len(chunks)}")

            async def process_all_chunks():
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} movies)")
                    
                    try:
                        if data_type == "letterboxd":
                            await get_movies(chunk, movies)
                        elif data_type == "poster":
                            await get_movie_posters(chunk, movies)
                        else:
                            await get_rich_data(chunk, movies, db, tmdb_key)
                            
                    except Exception as e:
                        logger.error(f"Error processing chunk {i+1}: {str(e)}")
                        continue
                        
                    # Small delay between chunks
                    if i < len(chunks) - 1:
                        await asyncio.sleep(1.0)

            # Run async processing with proper event loop handling
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context
                task = asyncio.create_task(process_all_chunks())
                loop.run_until_complete(task)
            else:
                # If we're in a sync context
                asyncio.run(process_all_chunks())
            
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

if __name__ == "__main__":
    main("letterboxd")
    main("poster")
    main("tmdb")