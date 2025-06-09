from fastapi import FastAPI, Request, HTTPException, status, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from typing import Union, Optional
from urllib.parse import urlparse, urlunparse
import pandas as pd
import traceback

from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.registry import DeferredJobRegistry

from worker import conn
from handle_recs import get_client_user_data, build_client_model

from config import config

# Setup logging
try:
    from data_processing.utils.logging_config import setup_logger
    logger = setup_logger('letterboxd.api')
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Setup validation
try:
    from data_processing.utils.validation import ValidationError
except ImportError:
    class ValidationError(Exception):
        pass

app = FastAPI(
    title="Letterboxd Recommendations API", 
    version="1.0.0",
    description="Movie recommendation system based on Letterboxd data using collaborative filtering"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=[
        "localhost", 
        "127.0.0.1", 
        "letterboxd.samlearner.com",
        "letterboxd-recommendations.herokuapp.com"
    ]
)

# CORS configuration
origins = [
    "http://localhost",
    "https://localhost", 
    "http://localhost:3000",
    "https://localhost:3000",
    "http://letterboxd-recommendations.herokuapp.com",
    "https://letterboxd-recommendations.herokuapp.com",
    "http://letterboxd.samlearner.com",
    "https://letterboxd.samlearner.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize queues
queue_pool = [Queue(channel, connection=conn) for channel in config.redis.queues]

class APIError(Exception):
    """Custom API error with status code"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation error", 
            "detail": str(exc),
            "type": "validation_error"
        }
    )

@app.exception_handler(APIError)
async def api_exception_handler(request: Request, exc: APIError):
    logger.error(f"API error on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "API error",
            "detail": exc.message,
            "type": "api_error"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "detail": "An unexpected error occurred",
            "type": "internal_error"
        }
    )

def validate_request_parameters(username: str, training_data_size: int, 
                              popularity_filter: int, data_opt_in: bool) -> tuple:
    """Validate and clean request parameters"""
    
    # Validate username
    if not username:
        raise APIError("Username is required")
    
    username = username.lower().strip()
    if not config.validate_username(username):
        raise APIError("Invalid username format. Only letters, numbers, underscores, and hyphens allowed.")
    
    # Validate training data size
    if training_data_size <= 0:
        raise APIError("Training data size must be positive")
    
    if training_data_size > config.model.max_training_size:
        raise APIError(f"Training data size cannot exceed {config.model.max_training_size}")
    
    # Validate popularity filter
    if popularity_filter < -1 or popularity_filter >= len(config.model.popularity_thresholds_500k):
        raise APIError(f"Popularity filter must be between -1 and {len(config.model.popularity_thresholds_500k) - 1}")
    
    # Validate data opt-in
    if not isinstance(data_opt_in, bool):
        raise APIError("data_opt_in must be a boolean value")
    
    return username, training_data_size, popularity_filter, data_opt_in

def get_queue_with_least_load() -> Queue:
    """Get the queue with the least load"""
    try:
        ordered_queues = sorted(
            queue_pool, 
            key=lambda queue: DeferredJobRegistry(queue=queue).count
        )
        
        queue_loads = [(q.name, DeferredJobRegistry(queue=q).count) for q in ordered_queues]
        logger.info(f"Queue loads: {queue_loads}")
        
        return ordered_queues[0]
    except Exception as e:
        logger.error(f"Error selecting queue: {str(e)}")
        # Fallback to first queue
        return queue_pool[0]

@app.get("/", response_class=HTMLResponse)
def homepage():
    """Redirect to new domain for backwards compatibility"""
    return RedirectResponse("https://letterboxd.samlearner.com", status_code=301)

@app.get("/get_recs")
def get_recs(
    username: str = Query(..., description="Letterboxd username"),
    training_data_size: int = Query(500000, description="Number of ratings to use for training"),
    popularity_filter: int = Query(-1, description="Popularity filter level (-1 for no filter)"),
    data_opt_in: bool = Query(False, description="Whether to store user data")
):
    """Generate movie recommendations for a user"""
    try:
        # Validate and clean parameters
        username, training_data_size, popularity_filter, data_opt_in = validate_request_parameters(
            username, training_data_size, popularity_filter, data_opt_in
        )
        
        # Set popularity threshold
        if popularity_filter >= 0:
            popularity_threshold = config.model.popularity_thresholds_500k[popularity_filter]
        else:
            popularity_threshold = None
        
        num_items = 2000
        
        # Select queue with least load
        q = get_queue_with_least_load()
        
        # Enqueue jobs with improved error handling
        try:
            job_get_user_data = q.enqueue(
                get_client_user_data,
                args=(username, data_opt_in),
                description=f"Scraping user data for {username} (sample: {training_data_size}, popularity_filter: {popularity_threshold}, data_opt_in: {data_opt_in})",
                result_ttl=config.redis.job_result_ttl,
                ttl=config.redis.job_ttl,
            )
            
            job_build_model = q.enqueue(
                build_client_model,
                args=(username, training_data_size, popularity_threshold, num_items),
                depends_on=job_get_user_data,
                description=f"Building model for {username} (sample: {training_data_size}, popularity_filter: {popularity_threshold})",
                result_ttl=config.redis.job_result_ttl,
                ttl=config.redis.job_ttl,
            )
            
        except Exception as e:
            logger.error(f"Error enqueuing jobs for {username}: {str(e)}")
            raise APIError("Failed to start recommendation generation", 500)
        
        logger.info(f"Enqueued jobs for user {username}: data={job_get_user_data.get_id()}, model={job_build_model.get_id()}")
        
        return JSONResponse({
            "status": "success",
            "message": "Recommendation generation started",
            "redis_get_user_data_job_id": job_get_user_data.get_id(),
            "redis_build_model_job_id": job_build_model.get_id(),
            "estimated_time_minutes": "2-5",
            "parameters": {
                "username": username,
                "training_data_size": training_data_size,
                "popularity_threshold": popularity_threshold,
                "data_opt_in": data_opt_in
            }
        })
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_recs for user {username}: {str(e)}")
        raise APIError("Failed to start recommendation generation", 500)

@app.get("/results")
def get_results(
    redis_build_model_job_id: str = Query(..., description="Build model job ID"),
    redis_get_user_data_job_id: str = Query(..., description="Get user data job ID")
):
    """Get results of recommendation generation"""
    try:
        # Validate job IDs
        if not redis_build_model_job_id or not redis_get_user_data_job_id:
            raise APIError("Both job IDs are required")
        
        if not isinstance(redis_build_model_job_id, str) or not isinstance(redis_get_user_data_job_id, str):
            raise APIError("Job IDs must be strings")
        
        job_ids = {
            "redis_build_model_job_id": redis_build_model_job_id,
            "redis_get_user_data_job_id": redis_get_user_data_job_id,
        }
        
        job_statuses = {}
        for key, job_id in job_ids.items():
            try:
                job = Job.fetch(job_id, connection=conn)
                job_statuses[key.replace("_id", "_status")] = job.get_status()
                
                # Check for job failures
                if job.is_failed:
                    error_info = {
                        "error": str(job.exc_info) if job.exc_info else "Unknown error",
                        "failed_at": job.ended_at.isoformat() if job.ended_at else None
                    }
                    job_statuses[key.replace("_id", "_error")] = error_info
                    
            except NoSuchJobError:
                job_statuses[key.replace("_id", "_status")] = "finished"
        
        # Get execution data
        execution_data = {"build_model_stage": "unknown"}
        try:
            end_job = Job.fetch(job_ids["redis_build_model_job_id"], connection=conn)
            execution_data = {
                "build_model_stage": end_job.meta.get("stage", "unknown"),
                "job_created_at": end_job.created_at.isoformat() if end_job.created_at else None,
                "job_started_at": end_job.started_at.isoformat() if end_job.started_at else None
            }
            
            # Add error information if job failed
            if end_job.is_failed:
                execution_data["error"] = end_job.meta.get("error", "Unknown error")
                
        except NoSuchJobError:
            execution_data = {"build_model_stage": "finished"}
        
        # Get user data
        try:
            user_job = Job.fetch(job_ids["redis_get_user_data_job_id"], connection=conn)
            execution_data.update({
                "num_user_ratings": user_job.meta.get("num_user_ratings", 0),
                "user_watchlist": user_job.meta.get("user_watchlist", []),
                "user_status": user_job.meta.get("user_status", "unknown")
            })
        except NoSuchJobError:
            pass
        
        # Return results
        try:
            end_job = Job.fetch(job_ids["redis_build_model_job_id"], connection=conn)
            
            if end_job.is_finished and not end_job.is_failed:
                logger.info(f"Recommendations completed successfully for job {redis_build_model_job_id}")
                
                result = end_job.result if end_job.result else []
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "completed",
                        "statuses": job_statuses,
                        "execution_data": execution_data,
                        "result": result,
                        "total_recommendations": len(result) if result else 0
                    }
                )
            elif end_job.is_failed:
                logger.error(f"Job {redis_build_model_job_id} failed")
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "failed",
                        "statuses": job_statuses,
                        "execution_data": execution_data,
                        "error": execution_data.get("error", "Job failed")
                    }
                )
            else:
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "processing",
                        "statuses": job_statuses,
                        "execution_data": execution_data,
                        "message": f"Currently at stage: {execution_data.get('build_model_stage', 'unknown')}"
                    }
                )
        except NoSuchJobError:
            # Job is finished but not in Redis anymore
            logger.info(f"Job {redis_build_model_job_id} completed but no longer in Redis")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "completed",
                    "statuses": job_statuses,
                    "execution_data": execution_data,
                    "result": [],
                    "message": "Job completed but results expired"
                }
            )
            
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error in get_results: {str(e)}")
        raise APIError("Failed to retrieve results", 500)

@app.get("/health")
def health_check():
    """Enhanced health check endpoint"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": pd.Timestamp.now().isoformat(),
            "version": "1.0.0"
        }
        
        # Test Redis connection
        try:
            conn.ping()
            health_data["redis"] = "connected"
            
            # Get queue statistics
            queue_stats = {}
            for queue in queue_pool:
                try:
                    queue_stats[queue.name] = {
                        "queued": len(queue),
                        "deferred": DeferredJobRegistry(queue=queue).count,
                        "failed": queue.failed_job_registry.count
                    }
                except Exception as e:
                    queue_stats[queue.name] = {"error": str(e)}
                    
            health_data["queues"] = queue_stats
            
        except Exception as e:
            health_data["redis"] = f"error: {str(e)}"
            health_data["status"] = "unhealthy"
        
        # Test database connection
        try:
            from data_processing.db_connect import get_db_manager
            db_manager = get_db_manager()
            db_manager.client.admin.command('ping')
            health_data["database"] = "connected"
        except Exception as e:
            health_data["database"] = f"error: {str(e)}"
            health_data["status"] = "unhealthy"
        
        # Test file availability
        try:
            import os
            required_files = [
                'data_processing/data/training_data.csv',
                'static/data/movie_data.csv',
                'data_processing/models/threshold_movie_list.txt'
            ]
            
            file_status = {}
            for file_path in required_files:
                file_status[file_path] = os.path.exists(file_path)
                if not os.path.exists(file_path):
                    health_data["status"] = "degraded"
            
            health_data["required_files"] = file_status
            
        except Exception as e:
            health_data["file_check"] = f"error: {str(e)}"
        
        status_code = 200 if health_data["status"] == "healthy" else (503 if health_data["status"] == "unhealthy" else 200)
        
        return JSONResponse(status_code=status_code, content=health_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )

@app.get("/stats")
def get_system_stats():
    """Get system statistics"""
    try:
        stats = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "max_training_size": config.model.max_training_size,
                "default_training_size": config.model.default_training_size,
                "scraping_delay": config.scraping.request_delay,
                "max_retries": config.scraping.max_retries
            }
        }
        
        # Add queue statistics
        total_queued = 0
        total_failed = 0
        for queue in queue_pool:
            try:
                queued = len(queue)
                failed = queue.failed_job_registry.count
                total_queued += queued
                total_failed += failed
            except:
                pass
        
        stats["queues"] = {
            "total_queued": total_queued,
            "total_failed": total_failed
        }
        
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise APIError("Failed to get system statistics", 500)

# Log configuration on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Letterboxd Recommendations API")
    config.log_configuration()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)