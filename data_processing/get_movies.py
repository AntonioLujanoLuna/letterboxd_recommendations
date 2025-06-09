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

async def fetch_letterboxd(url, session, input_data={}):
    async with session.get(url) as r:
        response = await r.read()

        # Parse ratings page response for each rating/review, use lxml parser for speed
        soup = BeautifulSoup(response, "lxml")
        # rating = review.find("span", attrs={"class": "rating"})
        
        movie_header = soup.find('section', attrs={'id': 'featured-film-header'})

        try:
            movie_title = movie_header.find('h1').text
        except AttributeError:
            movie_title = ''

        try:
            year = int(movie_header.find('small', attrs={'class': 'number'}).find('a').text)
        except AttributeError:
            year = None

        soup.find("span", attrs={"class": "rating"})

        try:
            imdb_link = soup.find("a", attrs={"data-track-action": "IMDb"})['href']
            imdb_id = imdb_link.split('/title')[1].strip('/').split('/')[0]
        except:
            imdb_link = ''
            imdb_id = ''

        try:
            tmdb_link = soup.find("a", attrs={"data-track-action": "TMDb"})['href']
            tmdb_id = tmdb_link.split('/movie')[1].strip('/').split('/')[0]
        except:
            tmdb_link = ''
            tmdb_id = ''
        
        movie_object = {
                    "movie_id": input_data["movie_id"],
                    "movie_title": movie_title,
                    "year_released": year,
                    "imdb_link": imdb_link,
                    "tmdb_link": tmdb_link,
                    "imdb_id": imdb_id,
                    "tmdb_id": tmdb_id
                }

        update_operation = UpdateOne({
                "movie_id": input_data["movie_id"]
            },
            {
                "$set": movie_object
            }, upsert=True)


        return update_operation


async def fetch_poster(url, session, input_data={}):
    async with session.get(url) as r:
        response = await r.read()

        # Parse poster standalone page
        soup = BeautifulSoup(response, "lxml")

        try:
            image_url = soup.find('div', attrs={'class': 'film-poster'}).find('img')['src'].split('?')[0]
            print(image_url)
            image_url = image_url.replace('https://a.ltrbxd.com/resized/', '').split('.jpg')[0]
            if 'https://s.ltrbxd.com/static/img/empty-poster' in image_url:
                image_url = ''
        except AttributeError:
            image_url = ''

        print(image_url)
        
        movie_object = {
                    "movie_id": input_data["movie_id"],
                }

        if image_url != "":
            movie_object["image_url"] = image_url
        
        movie_object['last_updated'] = datetime.datetime.now()

        update_operation = UpdateOne({
                "movie_id": input_data["movie_id"]
            },
            {
                "$set": movie_object
            }, upsert=True)

        return update_operation


async def fetch_tmdb_data(url, session, movie_data, input_data={}):
    async with session.get(url) as r:
        response = await r.json()

        movie_object = movie_data

        object_fields = ["genres", "production_countries", "spoken_languages"]
        for field_name in object_fields:
            try:
                movie_object[field_name] = [x["name"] for x in response[field_name]]
            except:
                movie_object[field_name] = None
        
        simple_fields = ["popularity", "overview", "runtime", "vote_average", "vote_count", "release_date", "original_language"]
        for field_name in simple_fields:
            try:
                movie_object[field_name] = response[field_name]
            except:
                movie_object[field_name] = None
        
        movie_object['last_updated'] = datetime.datetime.now()

        update_operation = UpdateOne({
                "movie_id": input_data["movie_id"]
            },
            {
                "$set": movie_object
            }, upsert=True)


        return update_operation


async def get_movies(movie_list, db_cursor, mongo_db):
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
            db_collection.bulk_write(update_operations, ordered=False)
            logger.info(f"Successfully updated {len(update_operations)} movies in database")
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")


async def get_movie_posters(movie_list, db_cursor, mongo_db):
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
            db_collection.bulk_write(update_operations, ordered=False)
            logger.info(f"Successfully updated {len(update_operations)} movie posters")
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")

async def get_rich_data(movie_list, db_cursor, mongo_db, tmdb_key):
    base_url = "https://api.themoviedb.org/3/movie/{}?api_key={}"

    async with ClientSession() as session:
        tasks = []
        
        # Fix: Handle both movie objects and movie IDs
        processed_movies = []
        for movie in movie_list:
            if isinstance(movie, dict):
                # It's a movie object
                if movie.get('tmdb_id'):
                    processed_movies.append(movie)
            elif isinstance(movie, str):
                # It's a movie ID string - need to fetch from DB
                movie_doc = db_cursor.find_one({"movie_id": movie})
                if movie_doc and movie_doc.get('tmdb_id'):
                    processed_movies.append(movie_doc)
        
        # Make requests for movies with TMDB IDs
        for movie in processed_movies:
            task = asyncio.ensure_future(
                fetch_tmdb_data(
                    base_url.format(movie["tmdb_id"], tmdb_key), 
                    session, 
                    movie, 
                    {"movie_id": movie["movie_id"]}
                )
            )
            tasks.append(task)

        # Gather all responses
        upsert_operations = await asyncio.gather(*tasks)

    try:
        if len(upsert_operations) > 0:
            movies = mongo_db.movies
            movies.bulk_write(upsert_operations, ordered=False)
    except BulkWriteError as bwe:
        pprint(bwe.details)

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
    
async def process_tmdb_response(response_data: bytes, input_data: dict) -> Optional[UpdateOne]:
    """Process TMDB API response data"""
    try:
        import json
        response = json.loads(response_data.decode('utf-8'))

        movie_object = input_data.copy()

        # Extract object fields
        object_fields = ["genres", "production_countries", "spoken_languages"]
        for field_name in object_fields:
            try:
                movie_object[field_name] = [x["name"] for x in response[field_name]]
            except (KeyError, TypeError):
                movie_object[field_name] = None
        
        # Extract simple fields
        simple_fields = ["popularity", "overview", "runtime", "vote_average", "vote_count", "release_date", "original_language"]
        for field_name in simple_fields:
            try:
                movie_object[field_name] = response[field_name]
            except KeyError:
                movie_object[field_name] = None
        
        movie_object['last_updated'] = datetime.datetime.now()

        return UpdateOne(
            {"movie_id": input_data["movie_id"]},
            {"$set": movie_object},
            upsert=True
        )

    except Exception as e:
        logger.error(f"Error processing TMDB data for {input_data['movie_id']}: {str(e)}")
        return None

async def get_movies_with_rate_limiting(movie_list, db_cursor, mongo_db):
    """Get movies with proper rate limiting"""
    url = "https://letterboxd.com/film/{}/"
    rate_limiter = RateLimiter(requests_per_second=Config.REQUEST_DELAY)
    
    async with ClientSession() as session:
        tasks = []
        for movie in movie_list:
            task = asyncio.ensure_future(
                fetch_with_retry(
                    url.format(movie), 
                    session, 
                    rate_limiter,
                    input_data={"movie_id": movie}
                )
            )
            tasks.append(task)

        # Process responses with validation
        responses = await asyncio.gather(*tasks)
        upsert_operations = []
        
        for response_data, input_data in responses:
            if response_data:
                operation = await process_movie_response(response_data, input_data)
                if operation:
                    upsert_operations.append(operation)

    # Bulk write to database
    if upsert_operations:
        try:
            movies = mongo_db.movies
            movies.bulk_write(upsert_operations, ordered=False)
            logger.info(f"Successfully processed {len(upsert_operations)} movies")
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")

def main(data_type="letterboxd"):
    """Main function with improved error handling and rate limiting"""
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
                            await get_rich_data(chunk, movies, tmdb_key)
                            pass
                            
                    except Exception as e:
                        logger.error(f"Error processing chunk {i+1}: {str(e)}")
                        continue
                        
                    # Small delay between chunks
                    if i < len(chunks) - 1:
                        await asyncio.sleep(1.0)

            # Run async processing
            asyncio.run(process_all_chunks())
            
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

if __name__ == "__main__":
    main("letterboxd")
    main("poster")
    main("tmdb")