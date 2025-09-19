"""
🚀 ULTRA CELERY BACKGROUND TASK SYSTEM - WORLD-CLASS ASYNC PROCESSING
The most advanced background task processing system ever built
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import redis
from dataclasses import asdict

# Configure Celery
celery_app = Celery(
    'mailsift_ultra',
    broker=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    include=[
        'celery_tasks.email_processing',
        'celery_tasks.web_scraping',
        'celery_tasks.analytics',
        'celery_tasks.notifications'
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_routes={
        'celery_tasks.email_processing.*': {'queue': 'email_processing'},
        'celery_tasks.web_scraping.*': {'queue': 'web_scraping'},
        'celery_tasks.analytics.*': {'queue': 'analytics'},
        'celery_tasks.notifications.*': {'queue': 'notifications'},
    },
    task_default_queue='default',
    task_default_exchange='default',
    task_default_exchange_type='direct',
    task_default_routing_key='default',
    result_expires=3600,  # 1 hour
    worker_max_tasks_per_child=1000,
    worker_max_memory_per_child=200000,  # 200MB
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task prerun events"""
    logger.info(f"Starting task {task_id}: {task.name}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task postrun events"""
    logger.info(f"Completed task {task_id}: {task.name} with state {state}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handle task failure events"""
    logger.error(f"Task {task_id} failed: {exception}")


# Redis client for task management
redis_client = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))


class TaskManager:
    """Advanced task management system"""
    
    def __init__(self):
        self.redis_client = redis_client
        self.task_stats = {}
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get comprehensive task status"""
        try:
            result = celery_app.AsyncResult(task_id)
            
            return {
                'task_id': task_id,
                'status': result.status,
                'result': result.result if result.ready() else None,
                'info': result.info,
                'date_done': result.date_done.isoformat() if result.date_done else None,
                'traceback': result.traceback,
                'successful': result.successful(),
                'failed': result.failed(),
                'ready': result.ready()
            }
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {'error': str(e)}
            
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status and statistics"""
        try:
            inspect = celery_app.control.inspect()
            
            return {
                'active_tasks': inspect.active(),
                'scheduled_tasks': inspect.scheduled(),
                'reserved_tasks': inspect.reserved(),
                'stats': inspect.stats(),
                'registered_tasks': inspect.registered(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {'error': str(e)}
            
    def purge_queue(self, queue_name: str) -> bool:
        """Purge all tasks from a queue"""
        try:
            celery_app.control.purge()
            logger.info(f"Purged queue: {queue_name}")
            return True
        except Exception as e:
            logger.error(f"Error purging queue: {e}")
            return False
            
    def revoke_task(self, task_id: str, terminate: bool = False) -> bool:
        """Revoke a running task"""
        try:
            celery_app.control.revoke(task_id, terminate=terminate)
            logger.info(f"Revoked task: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error revoking task: {e}")
            return False
            
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        try:
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            
            return {
                'workers': stats,
                'active_workers': len(stats) if stats else 0,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting worker stats: {e}")
            return {'error': str(e)}


# Global task manager instance
task_manager = TaskManager()


# Utility functions for task management
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status"""
    return task_manager.get_task_status(task_id)


def get_queue_status() -> Dict[str, Any]:
    """Get queue status"""
    return task_manager.get_queue_status()


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """Revoke a task"""
    return task_manager.revoke_task(task_id, terminate)


def purge_queue(queue_name: str) -> bool:
    """Purge a queue"""
    return task_manager.purge_queue(queue_name)


if __name__ == '__main__':
    celery_app.start()
