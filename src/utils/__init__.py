"""
Trading Bot Utilities Package
---------------------------
This package provides utility functions and classes for logging, metrics tracking,
notifications, and performance monitoring used throughout the trading bot application.
"""

from typing import Dict, Optional, Any, List

# Version Information
__version__ = '1.0.0'

# Logger exports
from .logger import (
    setup_logger,
    get_logger,
    TradeLogger,
    TimingLogger,
    CustomFormatter
)

# Metrics exports
from .metrics import (
    MetricsManager,
    LatencyTimer,
    setup_metrics
)

# Notification exports
from .notifications import (
    NotificationManager,
    NotificationChannel,
    TelegramChannel,
    EmailChannel,
    DiscordChannel,
    RateLimiter
)

# Convenience functions
def initialize_logging(
    name: str = 'trading_bot',
    log_dir: str = 'logs'
) -> TradeLogger:
    """
    Initialize logging with default configuration.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    return setup_logger(name=name)

def initialize_metrics(
    port: Optional[int] = None,
    enable_prometheus: bool = True
) -> MetricsManager:
    """
    Initialize metrics tracking with default configuration.
    
    Args:
        port: Port for Prometheus metrics server
        enable_prometheus: Whether to start Prometheus server
        
    Returns:
        Configured metrics manager instance
    """
    return setup_metrics(port=port)

def initialize_notifications(
    config: Dict[str, Any],
    channels: Optional[List[str]] = None
) -> NotificationManager:
    """
    Initialize notification system with specified channels.
    
    Args:
        config: Notification configuration dictionary
        channels: List of channels to enable (default: all available)
        
    Returns:
        Configured notification manager instance
    """
    notification_manager = NotificationManager(config)
    if channels:
        notification_manager.enable_channels(channels)
    return notification_manager

def get_system_info() -> Dict[str, Any]:
    """
    Get system information and statistics.
    
    Returns:
        Dictionary containing system metrics
    """
    import psutil
    import platform
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'disk_usage': psutil.disk_usage('/').percent
    }

# Package exports
__all__ = [
    # Logger classes and functions
    'setup_logger',
    'get_logger',
    'TradeLogger',
    'TimingLogger',
    'CustomFormatter',
    
    # Metrics classes and functions
    'MetricsManager',
    'LatencyTimer',
    'setup_metrics',
    
    # Notification classes
    'NotificationManager',
    'NotificationChannel',
    'TelegramChannel',
    'EmailChannel',
    'DiscordChannel',
    'RateLimiter',
    
    # Convenience functions
    'initialize_logging',
    'initialize_metrics',
    'initialize_notifications',
    'get_system_info',
]
