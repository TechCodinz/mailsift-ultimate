"""
🚀 ULTRA ERROR HANDLING & LOGGING SYSTEM - WORLD-CLASS ERROR MANAGEMENT
The most advanced error handling and logging system ever built
"""

import logging
import traceback
import json
import time
import hashlib
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import functools
import os
import sys
from collections import defaultdict
import threading
import queue

# Advanced logging imports
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    NETWORK = "network"
    PROCESSING = "processing"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: Optional[str] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


@dataclass
class ErrorReport:
    """Comprehensive error report"""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    traceback: str
    context: ErrorContext
    metadata: Dict[str, Any]
    timestamp: str
    resolved: bool = False
    resolution_notes: Optional[str] = None


class UltraErrorHandler:
    """The most advanced error handling system ever built"""

    def __init__(self, app=None):
        self.app = app
        self.error_reports = {}
        self.error_stats = defaultdict(int)
        self.error_patterns = defaultdict(int)
        self.alert_thresholds = {
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.MEDIUM: 20,
            ErrorSeverity.LOW: 100
        }
        self.alert_cooldowns = defaultdict(float)
        self.error_queue = queue.Queue(maxsize=1000)
        self.background_processor = None
        self._setup_logging()
        self._setup_sentry()
        self._start_background_processor()

    def _setup_logging(self):
        """Setup advanced logging configuration"""
        # Configure structured logging if available
        if STRUCTLOG_AVAILABLE:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

        # Configure standard logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/ultra_errors.log', mode='a')
            ]
        )

        self.logger = logging.getLogger('ultra_error_handler')

    def _setup_sentry(self):
        """Setup Sentry for error tracking"""
        if SENTRY_AVAILABLE and os.environ.get('SENTRY_DSN'):
            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )

            sentry_sdk.init(
                dsn=os.environ['SENTRY_DSN'],
                integrations=[
                    FlaskIntegration(),
                    sentry_logging
                ],
                traces_sample_rate=float(os.environ.get('SENTRY_TRACES', '0.1')),
                environment=os.environ.get('FLASK_ENV', 'development'),
                release=os.environ.get('RELEASE_VERSION', 'unknown')
            )

    def _start_background_processor(self):
        """Start background error processing"""
        self.background_processor = threading.Thread(
            target=self._process_error_queue,
            daemon=True
        )
        self.background_processor.start()

    def _process_error_queue(self):
        """Process error queue in background"""
        while True:
            try:
                error_report = self.error_queue.get(timeout=1)
                self._analyze_error_pattern(error_report)
                self._check_alert_thresholds(error_report)
                self._store_error_report(error_report)
                self.error_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing error queue: {e}")

    def _analyze_error_pattern(self, error_report: ErrorReport):
        """Analyze error patterns for insights"""
        # Create pattern hash
        pattern_hash = hashlib.md5(
            f"{error_report.category.value}:{error_report.exception_type}:{error_report.message[:100]}".encode()
        ).hexdigest()

        self.error_patterns[pattern_hash] += 1

        # Check for pattern anomalies
        if self.error_patterns[pattern_hash] > 10:
            self._trigger_pattern_alert(pattern_hash, error_report)

    def _check_alert_thresholds(self, error_report: ErrorReport):
        """Check if error thresholds are exceeded"""
        severity = error_report.severity
        threshold = self.alert_thresholds.get(severity, 100)

        # Check cooldown
        now = time.time()
        if now - self.alert_cooldowns[severity.value] < 300:  # 5 minutes cooldown
            return

        # Check threshold
        if self.error_stats[severity.value] >= threshold:
            self._trigger_severity_alert(severity, error_report)
            self.alert_cooldowns[severity.value] = now

    def _trigger_pattern_alert(self, pattern_hash: str, error_report: ErrorReport):
        """Trigger pattern-based alert"""
        alert_data = {
            'type': 'pattern_alert',
            'pattern_hash': pattern_hash,
            'count': self.error_patterns[pattern_hash],
            'error_report': asdict(error_report),
            'timestamp': datetime.now().isoformat()
        }

        self.logger.critical(f"ERROR PATTERN ALERT: {json.dumps(alert_data)}")

    def _trigger_severity_alert(
            self,
            severity: ErrorSeverity,
            error_report: ErrorReport):
        """Trigger severity-based alert"""
        alert_data = {
            'type': 'severity_alert',
            'severity': severity.value,
            'count': self.error_stats[severity.value],
            'error_report': asdict(error_report),
            'timestamp': datetime.now().isoformat()
        }

        self.logger.critical(f"SEVERITY ALERT: {json.dumps(alert_data)}")

    def _store_error_report(self, error_report: ErrorReport):
        """Store error report"""
        self.error_reports[error_report.error_id] = error_report
        self.error_stats[error_report.severity.value] += 1

        # Cleanup old reports (keep last 1000)
        if len(self.error_reports) > 1000:
            oldest_keys = sorted(self.error_reports.keys())[:100]
            for key in oldest_keys:
                del self.error_reports[key]

    def _determine_severity(
            self,
            exception: Exception,
            context: ErrorContext) -> ErrorSeverity:
        """Determine error severity based on exception and context"""
        exception_type = type(exception).__name__

        # Critical errors
        if exception_type in ['SystemExit', 'KeyboardInterrupt', 'MemoryError']:
            return ErrorSeverity.CRITICAL

        # High severity errors
        if exception_type in ['ConnectionError', 'TimeoutError', 'DatabaseError']:
            return ErrorSeverity.HIGH

        # Medium severity errors
        if exception_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorSeverity.MEDIUM

        # Low severity errors
        if exception_type in ['KeyError', 'IndexError']:
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM

    def _determine_category(
            self,
            exception: Exception,
            context: ErrorContext) -> ErrorCategory:
        """Determine error category based on exception and context"""
        message = str(exception).lower()

        # Authentication errors
        if 'auth' in message or 'login' in message or 'token' in message:
            return ErrorCategory.AUTHENTICATION

        # Authorization errors
        if 'permission' in message or 'access' in message or 'forbidden' in message:
            return ErrorCategory.AUTHORIZATION

        # Rate limiting errors
        if 'rate limit' in message or 'too many' in message:
            return ErrorCategory.RATE_LIMIT

        # External API errors
        if 'api' in message or 'request' in message or 'http' in message:
            return ErrorCategory.EXTERNAL_API

        # Database errors
        if 'database' in message or 'sql' in message or 'connection' in message:
            return ErrorCategory.DATABASE

        # Network errors
        if 'network' in message or 'timeout' in message or 'connection' in message:
            return ErrorCategory.NETWORK

        # Validation errors
        if 'validation' in message or 'invalid' in message or 'format' in message:
            return ErrorCategory.VALIDATION

        return ErrorCategory.UNKNOWN

    def handle_error(self, exception: Exception, context: ErrorContext = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Handle error with comprehensive reporting"""
        try:
            # Generate unique error ID
            error_id = hashlib.md5(
                f"{time.time()}:{id(exception)}:{traceback.format_exc()}".encode()
            ).hexdigest()[:16]

            # Determine severity and category
            severity = self._determine_severity(exception, context or ErrorContext())
            category = self._determine_category(exception, context or ErrorContext())

            # Create error report
            error_report = ErrorReport(
                error_id=error_id,
                severity=severity,
                category=category,
                message=str(exception),
                exception_type=type(exception).__name__,
                traceback=traceback.format_exc(),
                context=context or ErrorContext(),
                metadata=metadata or {},
                timestamp=datetime.now().isoformat()
            )

            # Log error
            log_level = {
                ErrorSeverity.CRITICAL: logging.CRITICAL,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.LOW: logging.INFO
            }.get(severity, logging.ERROR)

            self.logger.log(log_level, f"Error {error_id}: {exception}")

            # Queue for background processing
            try:
                self.error_queue.put_nowait(error_report)
            except queue.Full:
                self.logger.warning("Error queue full, dropping error report")

            # Send to Sentry if available
            if SENTRY_AVAILABLE:
                with sentry_sdk.push_scope() as scope:
                    if context:
                        scope.set_context("error_context", asdict(context))
                    if metadata:
                        scope.set_extra("metadata", metadata)
                    scope.set_tag("error_id", error_id)
                    scope.set_tag("severity", severity.value)
                    scope.set_tag("category", category.value)
                    sentry_sdk.capture_exception(exception)

            return error_id

        except Exception as e:
            self.logger.critical(f"Failed to handle error: {e}")
            return "unknown"

    def error_handler(self, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      category: ErrorCategory = ErrorCategory.UNKNOWN,
                      metadata: Dict[str, Any] = None):
        """Decorator for automatic error handling"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Create context from function arguments
                    context = ErrorContext(
                        endpoint=func.__name__,
                        timestamp=datetime.now().isoformat()
                    )

                    # Handle error
                    error_id = self.handle_error(e, context, metadata)

                    # Re-raise with error ID
                    raise type(e)(f"{str(e)} (Error ID: {error_id})") from e

            return wrapper
        return decorator

    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            'total_errors': sum(self.error_stats.values()),
            'severity_breakdown': dict(self.error_stats),
            'category_breakdown': {
                category.value: sum(1 for report in self.error_reports.values()
                                    if report.category == category)
                for category in ErrorCategory
            },
            'top_patterns': dict(sorted(self.error_patterns.items(),
                                        key=lambda x: x[1], reverse=True)[:10]),
            'recent_errors': [
                asdict(report) for report in
                sorted(self.error_reports.values(),
                       key=lambda x: x.timestamp, reverse=True)[:10]
            ],
            'alert_thresholds': {
                severity.value: threshold
                for severity, threshold in self.alert_thresholds.items()
            }
        }

    def get_error_report(self, error_id: str) -> Optional[ErrorReport]:
        """Get specific error report by ID"""
        return self.error_reports.get(error_id)

    def resolve_error(self, error_id: str, resolution_notes: str = None):
        """Mark error as resolved"""
        if error_id in self.error_reports:
            self.error_reports[error_id].resolved = True
            self.error_reports[error_id].resolution_notes = resolution_notes

    def cleanup_old_errors(self, days: int = 7):
        """Cleanup errors older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        old_errors = [
            error_id for error_id, report in self.error_reports.items()
            if report.timestamp < cutoff_str
        ]

        for error_id in old_errors:
            del self.error_reports[error_id]

        self.logger.info(f"Cleaned up {len(old_errors)} old error reports")


# Global error handler instance
ultra_error_handler = UltraErrorHandler()

# Convenience decorators


def handle_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  category: ErrorCategory = ErrorCategory.UNKNOWN,
                  metadata: Dict[str, Any] = None):
    """Convenience decorator for error handling"""
    return ultra_error_handler.error_handler(severity, category, metadata)


def log_error(exception: Exception, context: ErrorContext = None,
              metadata: Dict[str, Any] = None) -> str:
    """Convenience function for logging errors"""
    return ultra_error_handler.handle_error(exception, context, metadata)
