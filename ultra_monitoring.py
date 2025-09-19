"""
🚀 ULTRA MONITORING & ALERTING SYSTEM - WORLD-CLASS MONITORING
The most advanced monitoring and alerting system ever built
"""

import time
import json
import threading
import psutil
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from collections import defaultdict, deque
import queue

# Monitoring imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    """System metric"""
    name: str
    value: float
    type: MetricType
    labels: Dict[str, str]
    timestamp: str
    unit: Optional[str] = None

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: str
    resolved: bool = False
    resolved_at: Optional[str] = None

@dataclass
class HealthCheck:
    """Health check result"""
    service: str
    status: str  # healthy, unhealthy, degraded
    response_time: float
    error_message: Optional[str] = None
    timestamp: str = None
    metadata: Dict[str, Any] = None

class UltraMonitoringSystem:
    """The most advanced monitoring system ever built"""

    def __init__(self, app=None):
        self.app = app
        self.metrics = {}
        self.alerts = {}
        self.health_checks = {}
        self.alert_rules = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_queue = queue.Queue()
        self.monitoring_thread = None
        self.redis_client = None
        self.prometheus_metrics = {}
        self._setup_redis()
        self._setup_prometheus()
        self._setup_alert_rules()
        self._start_monitoring()

    def _setup_redis(self):
        """Setup Redis for metrics storage"""
        if REDIS_AVAILABLE and os.environ.get('REDIS_URL'):
            try:
                self.redis_client = redis.from_url(os.environ['REDIS_URL'])
                self.redis_client.ping()
            except Exception as e:
                logging.warning(f"Redis setup failed: {e}")

    def _setup_prometheus(self):
        """Setup Prometheus metrics"""
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics = {
                'http_requests': prometheus_client.Counter(
                    'ultra_http_requests_total',
                    'Total HTTP requests',
                    ['method', 'endpoint', 'status']
                ),
                'http_duration': prometheus_client.Histogram(
                    'ultra_http_request_duration_seconds',
                    'HTTP request duration',
                    ['method', 'endpoint']
                ),
                'email_extractions': prometheus_client.Counter(
                    'ultra_email_extractions_total',
                    'Total email extractions',
                    ['source', 'status']
                ),
                'api_calls': prometheus_client.Counter(
                    'ultra_api_calls_total',
                    'Total API calls',
                    ['endpoint', 'method']
                ),
                'error_count': prometheus_client.Counter(
                    'ultra_errors_total',
                    'Total errors',
                    ['severity', 'category']
                ),
                'active_users': prometheus_client.Gauge(
                    'ultra_active_users',
                    'Active users'
                ),
                'system_cpu': prometheus_client.Gauge(
                    'ultra_system_cpu_percent',
                    'System CPU usage percentage'
                ),
                'system_memory': prometheus_client.Gauge(
                    'ultra_system_memory_percent',
                    'System memory usage percentage'
                ),
                'system_disk': prometheus_client.Gauge(
                    'ultra_system_disk_percent',
                    'System disk usage percentage'
                )
            }

    def _setup_alert_rules(self):
        """Setup default alert rules"""
        self.alert_rules = {
            'high_cpu_usage': {
                'metric': 'system_cpu_percent',
                'threshold': 80.0,
                'level': AlertLevel.WARNING,
                'title': 'High CPU Usage',
                'message': 'CPU usage is above 80%'
            },
            'high_memory_usage': {
                'metric': 'system_memory_percent',
                'threshold': 85.0,
                'level': AlertLevel.WARNING,
                'title': 'High Memory Usage',
                'message': 'Memory usage is above 85%'
            },
            'high_disk_usage': {
                'metric': 'system_disk_percent',
                'threshold': 90.0,
                'level': AlertLevel.CRITICAL,
                'title': 'High Disk Usage',
                'message': 'Disk usage is above 90%'
            },
            'high_error_rate': {
                'metric': 'error_rate_percent',
                'threshold': 5.0,
                'level': AlertLevel.ERROR,
                'title': 'High Error Rate',
                'message': 'Error rate is above 5%'
            },
            'low_response_time': {
                'metric': 'avg_response_time_ms',
                'threshold': 1000.0,
                'level': AlertLevel.WARNING,
                'title': 'Slow Response Time',
                'message': 'Average response time is above 1000ms'
            }
        }

    def _start_monitoring(self):
        """Start background monitoring"""
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Collect application metrics
                self._collect_app_metrics()

                # Check alert rules
                self._check_alert_rules()

                # Perform health checks
                self._perform_health_checks()

                # Store metrics
                self._store_metrics()

                # Process alerts
                self._process_alerts()

                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric('system_cpu_percent'
                cpu_percent, MetricType.GAUGE)

            # Memory usage
            memory = psutil.virtual_memory()
            self._record_metric('system_memory_percent'
                memory.percent, MetricType.GAUGE)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._record_metric('system_disk_percent'
                disk_percent, MetricType.GAUGE)

            # Network I/O
            network = psutil.net_io_counters()
            self._record_metric('network_bytes_sent'
                network.bytes_sent, MetricType.COUNTER)
            self._record_metric('network_bytes_recv'
                network.bytes_recv, MetricType.COUNTER)

            # Process count
            process_count = len(psutil.pids())
            self._record_metric('system_process_count'
                process_count, MetricType.GAUGE)

        except Exception as e:
            logging.error(f"System metrics collection error: {e}")

    def _collect_app_metrics(self):
        """Collect application-level metrics"""
        try:
            # Database connections
            if self.redis_client:
                try:
                    self.redis_client.ping()
                    self._record_metric('redis_connected', 1, MetricType.GAUGE)
                except Exception:
                    self._record_metric('redis_connected', 0, MetricType.GAUGE)

            # Active sessions (placeholder)
            active_sessions = len(self.metrics.get('active_sessions', {}))
            self._record_metric('active_sessions'
                active_sessions, MetricType.GAUGE)

            # API response times (placeholder)
            avg_response_time = self._calculate_avg_response_time()
            self._record_metric('avg_response_time_ms'
                avg_response_time, MetricType.GAUGE)

        except Exception as e:
            logging.error(f"App metrics collection error: {e}")

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        response_times = self.metric_history.get('response_time_ms', deque())
        if response_times:
            return sum(response_times) / len(response_times)
        return 0.0

    def _check_alert_rules(self):
        """Check alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            try:
                metric_name = rule['metric']
                threshold = rule['threshold']
                level = rule['level']

                # Get current metric value
                current_value = self._get_metric_value(metric_name)

                if current_value is not None:
                    # Check if threshold is exceeded
                    if current_value > threshold:
                        # Check if alert already exists
                        existing_alert = self._get_active_alert(rule_name)

                        if not existing_alert:
                            # Create new alert
                            alert = Alert(
                                alert_id=f"{rule_name}_{int(time.time())}",
                                level=level,
                                title=rule['title'],
                                message=rule['message'],
                                metric_name=metric_name,
                                threshold=threshold,
                                current_value=current_value,
                                timestamp=datetime.now().isoformat()
                            )

                            self.alerts[alert.alert_id] = alert
                            self.alert_queue.put(alert)

                    else:
                        # Resolve existing alert
                            threshold is no longer exceeded
                        existing_alert = self._get_active_alert(rule_name)
                        if existing_alert:
                            existing_alert.resolved = True
                            existing_alert.resolved_at
                                datetime.now().isoformat()

            except Exception as e:
                logging.error(f"Alert rule check error for {rule_name}: {e}")

    def _get_active_alert(self, rule_name: str) -> Optional[Alert]:
        """Get active alert for a rule"""
        for alert in self.alerts.values():
            if not alert.resolved and rule_name in alert.alert_id:
                return alert
        return None

    def _perform_health_checks(self):
        """Perform health checks on various services"""
        health_checks = [
            self._check_database_health,
            self._check_redis_health,
            self._check_api_health,
            self._check_external_services_health
        ]

        for check_func in health_checks:
            try:
                check_func()
            except Exception as e:
                logging.error(f"Health check error: {e}")

    def _check_database_health(self):
        """Check database health"""
        # Placeholder for database health check
        health_check = HealthCheck(
            service='database',
            status='healthy',
            response_time=0.0,
            timestamp=datetime.now().isoformat()
        )
        self.health_checks['database'] = health_check

    def _check_redis_health(self):
        """Check Redis health"""
        start_time = time.time()
        try:
            if self.redis_client:
                self.redis_client.ping()
                status = 'healthy'
                error_message = None
            else:
                status = 'unhealthy'
                error_message = 'Redis client not available'
        except Exception as e:
            status = 'unhealthy'
            error_message = str(e)

        response_time = (time.time() - start_time) * 1000

        health_check = HealthCheck(
            service='redis',
            status=status,
            response_time=response_time,
            error_message=error_message,
            timestamp=datetime.now().isoformat()
        )
        self.health_checks['redis'] = health_check

    def _check_api_health(self):
        """Check API health"""
        start_time = time.time()
        try:
            # Check if app is responding
            if self.app:
                with self.app.test_client() as client:
                    response = client.get('/health')
                    if response.status_code == 200:
                        status = 'healthy'
                        error_message = None
                    else:
                        status = 'degraded'
                        error_message = f'HTTP {response.status_code}'
            else:
                status = 'unhealthy'
                error_message = 'App not available'
        except Exception as e:
            status = 'unhealthy'
            error_message = str(e)

        response_time = (time.time() - start_time) * 1000

        health_check = HealthCheck(
            service='api',
            status=status,
            response_time=response_time,
            error_message=error_message,
            timestamp=datetime.now().isoformat()
        )
        self.health_checks['api'] = health_check

    def _check_external_services_health(self):
        """Check external services health"""
        services = [
            {'name': 'stripe', 'url': 'https://api.stripe.com/v1/charges'},
            {'name': 'openai', 'url': 'https://api.openai.com/v1/models'}
        ]

        for service in services:
            start_time = time.time()
            try:
                response = requests.get(service['url'], timeout=5)
                if response.status_code < 400:
                    status = 'healthy'
                    error_message = None
                else:
                    status = 'degraded'
                    error_message = f'HTTP {response.status_code}'
            except Exception as e:
                status = 'unhealthy'
                error_message = str(e)

            response_time = (time.time() - start_time) * 1000

            health_check = HealthCheck(
                service=service['name'],
                status=status,
                response_time=response_time,
                error_message=error_message,
                timestamp=datetime.now().isoformat()
            )
            self.health_checks[service['name']] = health_check

    def _store_metrics(self):
        """Store metrics in Redis"""
        if self.redis_client:
            try:
                for metric_name, metric in self.metrics.items():
                    key = f"metrics:{metric_name}"
                    data = json.dumps(asdict(metric))
                    self.redis_client.setex(key, 3600, data)  # 1 hour TTL
            except Exception as e:
                logging.error(f"Metrics storage error: {e}")

    def _process_alerts(self):
        """Process alerts in the queue"""
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                self._send_alert(alert)
                self.alert_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Alert processing error: {e}")

    def _send_alert(self, alert: Alert):
        """Send alert via configured channels"""
        try:
            # Log alert
            logging.warning(f"ALERT: {alert.title} - {alert.message}")

            # Send email alert if configured
            if os.environ.get('ALERT_EMAIL'):
                self._send_email_alert(alert)

            # Send webhook alert if configured
            if os.environ.get('ALERT_WEBHOOK_URL'):
                self._send_webhook_alert(alert)

        except Exception as e:
            logging.error(f"Alert sending error: {e}")

    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.environ.get('SMTP_PORT', '587'))
            smtp_username = os.environ.get('SMTP_USERNAME')
            smtp_password = os.environ.get('SMTP_PASSWORD')
            alert_email = os.environ.get('ALERT_EMAIL')

            if not all([smtp_username, smtp_password, alert_email]):
                return

            msg = MIMEMultipart()
            msg['From'] = smtp_username
            msg['To'] = alert_email
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"

            body = f"""
            Alert: {alert.title}
            Level: {alert.level.value.upper()}
            Message: {alert.message}
            Metric: {alert.metric_name}
            Current Value: {alert.current_value}
            Threshold: {alert.threshold}
            Timestamp: {alert.timestamp}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            server.quit()

        except Exception as e:
            logging.error(f"Email alert error: {e}")

    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        try:
            webhook_url = os.environ.get('ALERT_WEBHOOK_URL')
            payload = {
                'alert_id': alert.alert_id,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

        except Exception as e:
            logging.error(f"Webhook alert error: {e}")

    def _record_metric(self, name: str, value: float, metric_type: MetricType,
                      labels: Dict[str, str] = None, unit: str = None):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            type=metric_type,
            labels=labels or {},
            timestamp=datetime.now().isoformat(),
            unit=unit
        )

        self.metrics[name] = metric
        self.metric_history[name].append(value)

        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and name in self.prometheus_metrics:
            prom_metric = self.prometheus_metrics[name]
            if metric_type == MetricType.COUNTER:
                prom_metric.inc(value)
            elif metric_type == MetricType.GAUGE:
                prom_metric.set(value)

    def _get_metric_value(self, name: str) -> Optional[float]:
        """Get current metric value"""
        if name in self.metrics:
            return self.metrics[name].value
        return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                name: {**asdict(metric), 'type': metric.type.value}
                for name, metric in self.metrics.items()
            },
            'alerts': {
                'active': [
                    {**asdict(alert), 'level': alert.level.value}
                    for alert in self.alerts.values() if not alert.resolved
                ],
                'resolved': [
                    {**asdict(alert), 'level': alert.level.value}
                    for alert in self.alerts.values() if alert.resolved
                ]
            },
            'health_checks': {name: asdict(check) for name
                check in self.health_checks.items()},
            'system_info': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'process_count': len(psutil.pids())
            }
        }

    def get_metrics_history(self
        metric_name: str, hours: int = 24) -> List[float]:
        """Get metric history for specified hours"""
        history = self.metric_history.get(metric_name, deque())
        return list(history)

    def add_alert_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """Add custom alert rule"""
        self.alert_rules[rule_name] = rule_config

    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now().isoformat()


# Global monitoring instance
ultra_monitoring = UltraMonitoringSystem()

# Convenience functions
def record_metric(name: str
    value: float, metric_type: MetricType = MetricType.GAUGE,
                 labels: Dict[str, str] = None, unit: str = None):
    """Convenience function for recording metrics"""
    ultra_monitoring._record_metric(name, value, metric_type, labels, unit)

def increment_counter(name: str
    value: float = 1, labels: Dict[str, str] = None):
    """Convenience function for incrementing counters"""
    ultra_monitoring._record_metric(name, value, MetricType.COUNTER, labels)

def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """Convenience function for setting gauge values"""
    ultra_monitoring._record_metric(name, value, MetricType.GAUGE, labels)
