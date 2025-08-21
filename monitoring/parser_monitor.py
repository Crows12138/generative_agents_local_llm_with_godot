#!/usr/bin/env python3
"""
Parser Monitoring and Alerting System
Automated monitoring for action parser with metrics and alerts
"""

import time
import json
import logging
import threading
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import psutil

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ParserMetrics:
    """Parser performance metrics"""
    total_requests: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    total_parse_time: float = 0.0
    average_parse_time: float = 0.0
    min_parse_time: float = float('inf')
    max_parse_time: float = 0.0
    parse_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Error distribution
    error_types: Dict[str, int] = field(default_factory=dict)
    error_messages: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Parse type metrics
    parse_type_counts: Dict[str, int] = field(default_factory=dict)
    parse_type_success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Confidence metrics
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    average_confidence: float = 0.0
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.successful_parses + self.failed_parses
        return self.successful_parses / total if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        return 1.0 - self.success_rate
    
    def update_parse_time(self, parse_time: float):
        """Update parse time metrics"""
        self.parse_times.append(parse_time)
        self.total_parse_time += parse_time
        self.total_requests += 1
        
        if self.total_requests > 0:
            self.average_parse_time = self.total_parse_time / self.total_requests
        
        self.min_parse_time = min(self.min_parse_time, parse_time)
        self.max_parse_time = max(self.max_parse_time, parse_time)
        self.last_updated = datetime.now()
    
    def update_success(self, parse_type: str, confidence: float = None):
        """Update success metrics"""
        self.successful_parses += 1
        self.parse_type_counts[parse_type] = self.parse_type_counts.get(parse_type, 0) + 1
        
        if confidence is not None:
            self.confidence_scores.append(confidence)
            if self.confidence_scores:
                self.average_confidence = statistics.mean(self.confidence_scores)
        
        self._update_parse_type_success_rate(parse_type, True)
        self.last_updated = datetime.now()
    
    def update_failure(self, parse_type: str, error_type: str, error_message: str):
        """Update failure metrics"""
        self.failed_parses += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        self.error_messages.append({
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "parse_type": parse_type
        })
        
        self._update_parse_type_success_rate(parse_type, False)
        self.last_updated = datetime.now()
    
    def _update_parse_type_success_rate(self, parse_type: str, success: bool):
        """Update success rate for specific parse type"""
        # Simple tracking - could be enhanced with more sophisticated metrics
        if parse_type not in self.parse_type_success_rates:
            self.parse_type_success_rates[parse_type] = 1.0 if success else 0.0
        else:
            # Simple exponential moving average
            alpha = 0.1
            current_rate = self.parse_type_success_rates[parse_type]
            new_value = 1.0 if success else 0.0
            self.parse_type_success_rates[parse_type] = alpha * new_value + (1 - alpha) * current_rate
    
    def get_recent_parse_times(self, seconds: int = 60) -> List[float]:
        """Get parse times from recent time window"""
        # For simplicity, return recent items from deque
        # In production, would need timestamps for accurate time windows
        return list(self.parse_times)[-min(len(self.parse_times), seconds):]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_requests": self.total_requests,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "average_parse_time": self.average_parse_time,
            "min_parse_time": self.min_parse_time if self.min_parse_time != float('inf') else 0,
            "max_parse_time": self.max_parse_time,
            "average_confidence": self.average_confidence,
            "error_types": dict(self.error_types),
            "parse_type_counts": dict(self.parse_type_counts),
            "parse_type_success_rates": dict(self.parse_type_success_rates),
            "last_updated": self.last_updated.isoformat()
        }

@dataclass
class Alert:
    """Alert configuration and instance"""
    name: str
    level: AlertLevel
    condition: str
    threshold: float
    message: str
    enabled: bool = True
    triggered: bool = False
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    cooldown_seconds: int = 300  # 5 minutes default
    
    def can_trigger(self) -> bool:
        """Check if alert can be triggered (respects cooldown)"""
        if not self.enabled:
            return False
        
        if self.last_triggered is None:
            return True
        
        return (datetime.now() - self.last_triggered).total_seconds() > self.cooldown_seconds
    
    def trigger(self):
        """Trigger the alert"""
        self.triggered = True
        self.last_triggered = datetime.now()
        self.trigger_count += 1

class ParserMonitor:
    """Main parser monitoring system"""
    
    def __init__(self, 
                 log_file: str = "parser_monitor.log",
                 metrics_file: str = "parser_metrics.json",
                 alerts_config_file: str = "alerts_config.json"):
        
        self.metrics = ParserMetrics()
        self.log_file = Path(log_file)
        self.metrics_file = Path(metrics_file)
        self.alerts_config_file = Path(alerts_config_file)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Alert system
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds
        
        # Load configuration
        self._load_default_alerts()
        self._load_alerts_config()
        
        self.logger.info("Parser monitor initialized")
    
    def _load_default_alerts(self):
        """Load default alert configurations"""
        default_alerts = [
            Alert(
                name="high_error_rate",
                level=AlertLevel.ERROR,
                condition="error_rate > threshold",
                threshold=0.10,  # 10% error rate
                message="Parser error rate is above 10%: {error_rate:.2%}",
                cooldown_seconds=300
            ),
            Alert(
                name="slow_parsing",
                level=AlertLevel.WARNING,
                condition="average_parse_time > threshold",
                threshold=1.0,  # 1 second
                message="Average parse time is slow: {average_parse_time:.3f}s",
                cooldown_seconds=180
            ),
            Alert(
                name="very_slow_parsing",
                level=AlertLevel.ERROR,
                condition="average_parse_time > threshold",
                threshold=5.0,  # 5 seconds
                message="Average parse time is very slow: {average_parse_time:.3f}s",
                cooldown_seconds=300
            ),
            Alert(
                name="low_confidence",
                level=AlertLevel.WARNING,
                condition="average_confidence < threshold",
                threshold=0.6,  # 60% confidence
                message="Average confidence is low: {average_confidence:.2%}",
                cooldown_seconds=600
            ),
            Alert(
                name="parser_failures",
                level=AlertLevel.CRITICAL,
                condition="failed_parses > threshold in last minute",
                threshold=10,  # 10 failures per minute
                message="High number of parser failures: {failed_parses} in recent period",
                cooldown_seconds=120
            ),
            Alert(
                name="memory_usage",
                level=AlertLevel.WARNING,
                condition="memory_usage > threshold",
                threshold=500,  # 500 MB
                message="High memory usage: {memory_usage:.1f} MB",
                cooldown_seconds=600
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.name] = alert
    
    def _load_alerts_config(self):
        """Load alerts configuration from file"""
        if self.alerts_config_file.exists():
            try:
                with open(self.alerts_config_file, 'r') as f:
                    config = json.load(f)
                
                for alert_name, alert_config in config.get("alerts", {}).items():
                    if alert_name in self.alerts:
                        alert = self.alerts[alert_name]
                        alert.threshold = alert_config.get("threshold", alert.threshold)
                        alert.enabled = alert_config.get("enabled", alert.enabled)
                        alert.cooldown_seconds = alert_config.get("cooldown_seconds", alert.cooldown_seconds)
                
                self.monitoring_interval = config.get("monitoring_interval", self.monitoring_interval)
                
                self.logger.info(f"Loaded alerts configuration from {self.alerts_config_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to load alerts config: {e}")
    
    def _save_alerts_config(self):
        """Save current alerts configuration"""
        config = {
            "monitoring_interval": self.monitoring_interval,
            "alerts": {}
        }
        
        for alert_name, alert in self.alerts.items():
            config["alerts"][alert_name] = {
                "threshold": alert.threshold,
                "enabled": alert.enabled,
                "cooldown_seconds": alert.cooldown_seconds
            }
        
        try:
            with open(self.alerts_config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save alerts config: {e}")
    
    def record_parse_start(self) -> float:
        """Record start of parsing operation"""
        return time.time()
    
    def record_parse_success(self, start_time: float, parse_type: str, confidence: float = None):
        """Record successful parsing operation"""
        parse_time = time.time() - start_time
        self.metrics.update_parse_time(parse_time)
        self.metrics.update_success(parse_type, confidence)
        
        self.logger.debug(f"Parse success: {parse_type}, time: {parse_time:.3f}s, confidence: {confidence}")
    
    def record_parse_failure(self, start_time: float, parse_type: str, error_type: str, error_message: str):
        """Record failed parsing operation"""
        parse_time = time.time() - start_time
        self.metrics.update_parse_time(parse_time)
        self.metrics.update_failure(parse_type, error_type, error_message)
        
        self.logger.warning(f"Parse failure: {parse_type}, error: {error_type}, message: {error_message}")
    
    def add_alert_handler(self, handler: Callable[[Alert, Dict[str, Any]], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self):
        """Check all alerts against current metrics"""
        metrics_dict = self.metrics.to_dict()
        
        # Add system metrics
        process = psutil.Process()
        metrics_dict["memory_usage"] = process.memory_info().rss / 1024 / 1024  # MB
        metrics_dict["cpu_percent"] = process.cpu_percent()
        
        for alert_name, alert in self.alerts.items():
            if not alert.can_trigger():
                continue
            
            should_trigger = self._evaluate_alert_condition(alert, metrics_dict)
            
            if should_trigger and not alert.triggered:
                alert.trigger()
                self._fire_alert(alert, metrics_dict)
            elif not should_trigger and alert.triggered:
                alert.triggered = False
    
    def _evaluate_alert_condition(self, alert: Alert, metrics: Dict[str, Any]) -> bool:
        """Evaluate if alert condition is met"""
        try:
            if alert.name == "high_error_rate":
                return metrics.get("error_rate", 0) > alert.threshold
            
            elif alert.name == "slow_parsing":
                return metrics.get("average_parse_time", 0) > alert.threshold
            
            elif alert.name == "very_slow_parsing":
                return metrics.get("average_parse_time", 0) > alert.threshold
            
            elif alert.name == "low_confidence":
                return metrics.get("average_confidence", 1.0) < alert.threshold
            
            elif alert.name == "parser_failures":
                # Check failures in recent time window
                recent_failures = len([msg for msg in self.metrics.error_messages 
                                     if (datetime.now() - datetime.fromisoformat(msg["timestamp"])).total_seconds() < 60])
                return recent_failures > alert.threshold
            
            elif alert.name == "memory_usage":
                return metrics.get("memory_usage", 0) > alert.threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating alert {alert.name}: {e}")
            return False
    
    def _fire_alert(self, alert: Alert, metrics: Dict[str, Any]):
        """Fire an alert"""
        try:
            message = alert.message.format(**metrics)
            
            self.logger.log(
                getattr(logging, alert.level.value.upper(), logging.INFO),
                f"ALERT [{alert.level.value.upper()}] {alert.name}: {message}"
            )
            
            # Call alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert, metrics)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to fire alert {alert.name}: {e}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started parser monitoring")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped parser monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self.check_alerts()
                self._save_metrics()
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _save_metrics(self):
        """Save current metrics to file"""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics.to_dict(),
                "alerts_status": {
                    name: {
                        "triggered": alert.triggered,
                        "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
                        "trigger_count": alert.trigger_count
                    }
                    for name, alert in self.alerts.items()
                }
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "metrics": self.metrics.to_dict(),
            "alerts": {
                name: {
                    "name": alert.name,
                    "level": alert.level.value,
                    "enabled": alert.enabled,
                    "triggered": alert.triggered,
                    "threshold": alert.threshold,
                    "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
                    "trigger_count": alert.trigger_count
                }
                for name, alert in self.alerts.items()
            },
            "system_info": {
                "monitoring_active": self.monitoring_active,
                "monitoring_interval": self.monitoring_interval,
                "uptime": (datetime.now() - self.metrics.last_updated).total_seconds()
            }
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = ParserMetrics()
        self.logger.info("Metrics reset")
    
    def configure_alert(self, alert_name: str, **kwargs):
        """Configure an alert"""
        if alert_name in self.alerts:
            alert = self.alerts[alert_name]
            for key, value in kwargs.items():
                if hasattr(alert, key):
                    setattr(alert, key, value)
            
            self._save_alerts_config()
            self.logger.info(f"Alert {alert_name} configured: {kwargs}")
        else:
            self.logger.error(f"Alert {alert_name} not found")

# Context manager for easy monitoring
class ParserMonitorContext:
    """Context manager for parser monitoring"""
    
    def __init__(self, monitor: ParserMonitor, parse_type: str):
        self.monitor = monitor
        self.parse_type = parse_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = self.monitor.record_parse_start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            self.monitor.record_parse_success(self.start_time, self.parse_type)
        else:
            # Failure
            error_type = exc_type.__name__ if exc_type else "UnknownError"
            error_message = str(exc_val) if exc_val else "Unknown error"
            self.monitor.record_parse_failure(self.start_time, self.parse_type, error_type, error_message)

# Default alert handlers
def console_alert_handler(alert: Alert, metrics: Dict[str, Any]):
    """Simple console alert handler"""
    print(f"🚨 [{alert.level.value.upper()}] {alert.name}: {alert.message.format(**metrics)}")

def email_alert_handler(alert: Alert, metrics: Dict[str, Any]):
    """Email alert handler (placeholder)"""
    # In production, implement actual email sending
    print(f"📧 Email Alert: {alert.name} - {alert.message.format(**metrics)}")

def webhook_alert_handler(alert: Alert, metrics: Dict[str, Any]):
    """Webhook alert handler (placeholder)"""
    # In production, implement actual webhook call
    print(f"🔗 Webhook Alert: {alert.name} - {alert.message.format(**metrics)}")

# Global monitor instance
_parser_monitor = None

def get_parser_monitor() -> ParserMonitor:
    """Get global parser monitor instance"""
    global _parser_monitor
    if _parser_monitor is None:
        _parser_monitor = ParserMonitor()
        
        # Add default alert handlers
        _parser_monitor.add_alert_handler(console_alert_handler)
        
        # Start monitoring
        _parser_monitor.start_monitoring()
    
    return _parser_monitor

# Test and demo
def test_parser_monitor():
    """Test parser monitoring system"""
    print("=== Parser Monitor Test ===\\n")
    
    monitor = ParserMonitor()
    
    # Add alert handlers
    monitor.add_alert_handler(console_alert_handler)
    
    # Simulate some parsing operations
    print("Simulating parsing operations...")
    
    for i in range(50):
        with ParserMonitorContext(monitor, "decision") as ctx:
            time.sleep(0.01)  # Simulate processing
            if i % 10 == 0:  # Simulate some failures
                raise ValueError(f"Simulated error {i}")
    
    # Check current metrics
    print("\\nCurrent metrics:")
    metrics = monitor.get_dashboard_data()
    print(json.dumps(metrics["metrics"], indent=2))
    
    # Trigger some alerts
    print("\\nTriggering alerts...")
    monitor.metrics.failed_parses = 20  # High failure count
    monitor.check_alerts()
    
    # Configure an alert
    print("\\nConfiguring alert...")
    monitor.configure_alert("high_error_rate", threshold=0.05)  # Lower threshold
    
    print("\\n=== Test completed ===")

if __name__ == "__main__":
    test_parser_monitor()