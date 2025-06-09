import os
import signal
import sys
from urllib.parse import urlparse
import redis
from rq import Worker, Queue, Connection
from rq.middleware import Middleware

from config import config

# Setup logging
try:
    from data_processing.utils.logging_config import setup_logger
    logger = setup_logger('letterboxd.worker')
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class LoggingMiddleware(Middleware):
    """Middleware to log job execution"""
    
    def call(self, queue, job_func, job, *args, **kwargs):
        logger.info(f"Starting job {job.id}: {job.description}")
        try:
            result = job_func(job, *args, **kwargs)
            logger.info(f"Completed job {job.id}")
            return result
        except Exception as e:
            logger.error(f"Job {job.id} failed: {str(e)}")
            raise

class MemoryMonitoringMiddleware(Middleware):
    """Middleware to monitor memory usage"""
    
    def call(self, queue, job_func, job, *args, **kwargs):
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Job {job.id} starting with {initial_memory:.1f} MB memory usage")
        except ImportError:
            initial_memory = None
        
        try:
            result = job_func(job, *args, **kwargs)
            
            if initial_memory:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = final_memory - initial_memory
                logger.info(f"Job {job.id} completed with {final_memory:.1f} MB memory usage (delta: {memory_diff:+.1f} MB)")
            
            return result
        except Exception as e:
            if initial_memory:
                try:
                    final_memory = process.memory_info().rss / 1024 / 1024  # MB
                    logger.error(f"Job {job.id} failed with {final_memory:.1f} MB memory usage")
                except:
                    pass
            raise

# Redis connection
redis_url = config.redis.url
conn = redis.from_url(redis_url)

# Test Redis connection
try:
    conn.ping()
    logger.info(f"Successfully connected to Redis at {redis_url}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    sys.exit(1)

# Create queues
listen = config.redis.queues
queues = [Queue(queue_name, connection=conn) for queue_name in listen]

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down worker gracefully...")
    sys.exit(0)

def create_worker():
    """Create and configure worker"""
    worker = Worker(
        queues, 
        connection=conn,
        name=f"letterboxd-worker-{os.getpid()}",
        log_job_description=True,
        job_monitoring_interval=30,  # Monitor jobs every 30 seconds
    )
    
    # Add middleware
    worker.push_middleware(LoggingMiddleware())
    worker.push_middleware(MemoryMonitoringMiddleware())
    
    return worker

if __name__ == '__main__':
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info(f"Starting worker listening to queues: {listen}")
    
    try:
        worker = create_worker()
        
        # Start worker with error handling
        with Connection(conn):
            worker.work(logging_level='INFO')
            
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed with error: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Worker shutdown complete")