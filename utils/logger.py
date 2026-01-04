"""
Logging configuration for the project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

def setup_logger(name=__name__, log_file=None, level=logging.INFO):
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Default log file name
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'app_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # JSON handler for structured logging
    json_handler = logging.FileHandler(log_file.with_name(f'{log_file.stem}_json.log'))
    json_handler.setLevel(level)
    
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                'timestamp': datetime.now().isoformat(),
                'logger': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if record.exc_info:
                log_obj['exception'] = self.formatException(record.exc_info)
            return json.dumps(log_obj)
    
    json_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_handler)
    
    return logger

def log_performance_metrics(logger, metrics_dict, phase="training"):
    """
    Log performance metrics in a structured way
    
    Args:
        logger: Logger instance
        metrics_dict: Dictionary of metrics
        phase: Phase of training/testing
    """
    metrics_str = "\n" + "="*50 + f"\n{phase.upper()} METRICS\n" + "="*50
    for metric, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            metrics_str += f"\n{metric.replace('_', ' ').title()}: {value:.4f}"
        else:
            metrics_str += f"\n{metric.replace('_', ' ').title()}: {value}"
    
    metrics_str += "\n" + "="*50
    logger.info(metrics_str)