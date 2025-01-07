"""
Logging configuration for the trading bot.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from typing import Optional, Dict, Any

class CustomFormatter(logging.Formatter):
    """Custom formatter with color support and JSON formatting."""
    
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def __init__(self, use_colors: bool = True, json_format: bool = False):
        super().__init__()
        self.use_colors = use_colors
        self.json_format = json_format

    def format(self, record: logging.LogRecord) -> str:
        # Add timestamp to the record
        record.timestamp = datetime.utcnow().isoformat()

        if self.json_format:
            log_data = {
                'timestamp': record.timestamp,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add extra fields if they exist
            if hasattr(record, 'extra_fields'):
                log_data.update(record.extra_fields)
                
            return json.dumps(log_data)
        
        # Format for console output
        log_message = f'[{record.timestamp}] {record.levelname:<8} {record.module}:{record.funcName}:{record.lineno} - {record.getMessage()}'
        
        if self.use_colors:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            log_message = f'{color}{log_message}{reset}'
            
        return log_message

class TradeLogger:
    """Custom logger for trading operations."""
    
    def __init__(self, name: str, log_dir: str = 'logs'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CustomFormatter(use_colors=True))
        self.logger.addHandler(console_handler)
        
        # File handler with JSON formatting for all logs
        file_handler = RotatingFileHandler(
            self.log_dir / 'trading.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(CustomFormatter(use_colors=False, json_format=True))
        self.logger.addHandler(file_handler)
        
        # Separate error log file
        error_handler = TimedRotatingFileHandler(
            self.log_dir / 'error.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(CustomFormatter(use_colors=False, json_format=True))
        self.logger.addHandler(error_handler)

    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a message with optional extra fields."""
        if extra:
            extra_record = logging.LogRecord(
                name=self.logger.name,
                level=getattr(logging, level.upper()),
                pathname='',
                lineno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            extra_record.extra_fields = extra
            for handler in self.logger.handlers:
                handler.emit(extra_record)
        else:
            getattr(self.logger, level.lower())(message)

def setup_logger(name: str = 'trading_bot') -> TradeLogger:
    """Setup and return a configured logger instance."""
    return TradeLogger(name)

def get_logger(name: str) -> TradeLogger:
    """Get or create a logger instance."""
    return setup_logger(name)

# Context manager for timing operations
class TimingLogger:
    """Context manager for timing operations."""
    
    def __init__(self, logger: TradeLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log('info', f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.log('info', 
                          f"Completed {self.operation}",
                          {'duration_ms': duration.total_seconds() * 1000})
        else:
            self.logger.log('error',
                          f"Failed {self.operation}",
                          {
                              'duration_ms': duration.total_seconds() * 1000,
                              'error': str(exc_val)
                          })
