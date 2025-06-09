from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from typing import Union
from urllib.parse import urlparse, urlunparse
import pandas as pd

from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.registry import DeferredJobRegistry

from worker import conn
from handle_recs import get_client_user_data, build_client_model

from config import Config

from data_processing.utils.logging_config import setup_logger
from data_processing.utils.validation import ValidationError

# Setup logging
logger = setup_logger('letterboxd.api')

app = FastAPI(title="Letterboxd Recommendations API", version="1.0.0")

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
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize queues
queue_pool = [Queue(channel, connection=conn) for channel in Config.REDIS_QUEUES]

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": "Validation error", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

@app.get("/", response_class=HTMLResponse)
def homepage():
    """Redirect to new domain for backwards compatibility"""
    return RedirectResponse("https://letterboxd.samlearner.com")

@app.get("/get_recs")
def get_recs(
    username: str, 
    training_data_size: int, 
    popularity_filter: int, 
    data_opt_in: bool
):
    """Generate movie recommendations for a user"""
    try:
        # Validate inputs
        if not Config.validate_username(username.lower().strip()):
            raise HTTPException(
                status_code=400, 
                detail="Invalid username format"
            )
        
        username = username.lower().strip()
        
        # Validate training data size
        if training_data_size <= 0 or training_data_size > Config.MAX_TRAINING_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Training data size must be between 1 and {Config.MAX_TRAINING_SIZE}"
            )
        
        # Validate popularity filter
        if popularity_filter < -1 or popularity_filter >= len(Config.POPULARITY_THRESHOLDS_500K):
            raise HTTPException(
                status_code=400,
                detail="Invalid popularity filter value"
            )
        
        # Set popularity threshold
        if popularity_filter >= 0:
            popularity_threshold = Config.POPULARITY_THRESHOLDS_500K[popularity_filter]
        else:
            popularity_threshold = None
        
        num_items = 2000
        
        # Select queue with least load
        ordered_queues = sorted(
            queue_pool, key=lambda queue: DeferredJobRegistry(queue=queue).count
        )
        
        logger.info(f"Queue loads: {[(q.name, DeferredJobRegistry(queue=q).count) for q in ordered_queues]}")
        q = ordered_queues[0]
        
        # Enqueue jobs
        job_get_user_data = q.enqueue(
            get_client_user_data,
            args=(username, data_opt_in),
            description=f"Scraping user data for {username} (sample: {training_data_size}, popularity_filter: {popularity_threshold}, data_opt_in: {data_opt_in})",
            result_ttl=Config.JOB_RESULT_TTL,
            ttl=Config.JOB_TTL,
        )
        
        job_build_model = q.enqueue(
            build_client_model,
            args=(username, training_data_size, popularity_threshold, num_items),
            depends_on=job_get_user_data,
            description=f"Building model for {username} (sample: {training_data_size}, popularity_filter: {popularity_threshold})",
            result_ttl=Config.JOB_RESULT_TTL,
            ttl=Config.JOB_TTL,
        )
        
        logger.info(f"Enqueued jobs for user {username}: data={job_get_user_data.get_id()}, model={job_build_model.get_id()}")
        
        return JSONResponse({
            "redis_get_user_data_job_id": job_get_user_data.get_id(),
            "redis_build_model_job_id": job_build_model.get_id(),
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_recs for user {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start recommendation generation")

@app.get("/results")
def get_results(redis_build_model_job_id: str, redis_get_user_data_job_id: str):
    """Get results of recommendation generation"""
    try:
        job_ids = {
            "redis_build_model_job_id": redis_build_model_job_id,
            "redis_get_user_data_job_id": redis_get_user_data_job_id,
        }
        
        # Validate job IDs
        for key, job_id in job_ids.items():
            if not job_id or not isinstance(job_id, str):
                raise HTTPException(status_code=400, detail=f"Invalid {key}")
        
        job_statuses = {}
        for key, job_id in job_ids.items():
            try:
                job = Job.fetch(job_id, connection=conn)
                job_statuses[key.replace("_id", "_status")] = job.get_status()
            except NoSuchJobError:
                job_statuses[key.replace("_id", "_status")] = "finished"
        
        # Get execution data
        try:
            end_job = Job.fetch(job_ids["redis_build_model_job_id"], connection=conn)
            execution_data = {"build_model_stage": end_job.meta.get("stage")}
        except NoSuchJobError:
            execution_data = {"build_model_stage": "finished"}
        
        # Get user data
        try:
            user_job = Job.fetch(job_ids["redis_get_user_data_job_id"], connection=conn)
            execution_data.update({
                "num_user_ratings": user_job.meta.get("num_user_ratings"),
                "user_watchlist": user_job.meta.get("user_watchlist"),
                "user_status": user_job.meta.get("user_status")
            })
        except NoSuchJobError:
            pass
        
        # Return results
        try:
            end_job = Job.fetch(job_ids["redis_build_model_job_id"], connection=conn)
            if end_job.is_finished:
                logger.info(f"Recommendations completed for job {redis_build_model_job_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "statuses": job_statuses,
                        "execution_data": execution_data,
                        "result": end_job.result
                    }
                )
            else:
                return JSONResponse(
                    status_code=202,
                    content={
                        "statuses": job_statuses,
                        "execution_data": execution_data
                    }
                )
        except NoSuchJobError:
            # Job is finished but not in Redis anymore
            return JSONResponse(
                status_code=200,
                content={
                    "statuses": job_statuses,
                    "execution_data": execution_data,
                    "result": []
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_results: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve results")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        conn.ping()
        
        return JSONResponse({
            "status": "healthy",
            "redis": "connected",
            "queues": [q.name for q in queue_pool]
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )