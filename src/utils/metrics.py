"""
Metrics tracking and monitoring for the trading bot.
"""

from typing import Dict, Optional
import time
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import threading
from .logger import get_logger

logger = get_logger(__name__)

class MetricsManager:
    """Manages trading metrics and monitoring."""
    
    def __init__(self, port: int = 8000):
        # Initialize Prometheus metrics
        self.initialize_metrics()
        
        # Start Prometheus HTTP server
        try:
            start_http_server(port)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            
        self.start_time = datetime.now()
        self._lock = threading.Lock()

    def initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        # Trading metrics
        self.trades_counter = Counter(
            'trading_bot_trades_total',
            'Total number of trades',
            ['outcome']  # success, failure
        )
        
        self.position_gauge = Gauge(
            'trading_bot_position_size',
            'Current position size',
            ['symbol']
        )
        
        self.balance_gauge = Gauge(
            'trading_bot_balance',
            'Current balance',
            ['asset']
        )
        
        self.profit_counter = Counter(
            'trading_bot_profit_total',
            'Total profit/loss',
            ['symbol']
        )
        
        # Performance metrics
        self.latency_histogram = Histogram(
            'trading_bot_operation_latency_seconds',
            'Operation latency in seconds',
            ['operation']
        )
        
        self.api_requests = Counter(
            'trading_bot_api_requests_total',
            'Total API requests',
            ['endpoint', 'status']
        )
        
        # Risk metrics
        self.drawdown_gauge = Gauge(
            'trading_bot_drawdown_percentage',
            'Current drawdown as percentage'
        )
        
        self.risk_exposure_gauge = Gauge(
            'trading_bot_risk_exposure',
            'Current risk exposure'
        )
        
        # System metrics
        self.memory_gauge = Gauge(
            'trading_bot_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage_gauge = Gauge(
            'trading_bot_cpu_usage_percentage',
            'CPU usage percentage'
        )

    def record_trade(self, symbol: str, outcome: str, profit: float, 
                    position_size: float):
        """Record trade metrics."""
        with self._lock:
            self.trades_counter.labels(outcome=outcome).inc()
            self.position_gauge.labels(symbol=symbol).set(position_size)
            self.profit_counter.labels(symbol=symbol).inc(profit)

    def record_balance(self, asset: str, balance: float):
        """Record balance metrics."""
        with self._lock:
            self.balance_gauge.labels(asset=asset).set(balance)

    def record_api_request(self, endpoint: str, status: str, 
                         latency: float):
        """Record API request metrics."""
        with self._lock:
            self.api_requests.labels(endpoint=endpoint, status=status).inc()
            self.latency_histogram.labels(operation=f"api_{endpoint}").observe(latency)

    def record_risk_metrics(self, drawdown: float, risk_exposure: float):
        """Record risk metrics."""
        with self._lock:
            self.drawdown_gauge.set(drawdown)
            self.risk_exposure_gauge.set(risk_exposure)

    def record_system_metrics(self, memory_usage: float, cpu_usage: float):
        """Record system resource metrics."""
        with self._lock:
            self.memory_gauge.set(memory_usage)
            self.cpu_usage_gauge.set(cpu_usage)

class LatencyTimer:
    """Context manager for measuring operation latency."""
    
    def __init__(self, metrics_manager: MetricsManager, operation: str):
        self.metrics_manager = metrics_manager
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        self.metrics_manager.latency_histogram.labels(
            operation=self.operation
        ).observe(latency)

def setup_metrics(port: Optional[int] = None) -> MetricsManager:
    """Setup and return a metrics manager instance."""
    if port is None:
        port = 8000
    return MetricsManager(port)
