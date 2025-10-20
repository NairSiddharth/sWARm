"""
Centralized logging configuration for new_pipeline.

This module provides a configured logger that should be used throughout
the new_pipeline for consistent logging behavior.

Usage:
    from new_pipeline.common.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Starting calculation")
    logger.debug("Detailed information: %s", data)
    logger.error("Error occurred", exc_info=True)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Create logs directory in new_pipeline folder
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Global configuration
DEFAULT_LOG_LEVEL = logging.DEBUG
CONSOLE_LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = logging.DEBUG

# Format strings
CONSOLE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
FILE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
DATE_FORMAT = '%H:%M:%S'
FILE_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    This function returns a logger configured with both console and file handlers.
    If the logger has already been configured, it returns the existing logger.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_file: Optional custom log file name (defaults to daily log)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Debug information: %s", variable)
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(DEFAULT_LOG_LEVEL)

        # Console handler for INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(CONSOLE_LOG_LEVEL)
        console_formatter = logging.Formatter(
            CONSOLE_FORMAT,
            datefmt=DATE_FORMAT
        )
        console_handler.setFormatter(console_formatter)

        # File handler for DEBUG and above
        if log_file is None:
            log_file = f"new_pipeline_{datetime.now():%Y%m%d}.log"

        log_path = LOG_DIR / log_file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(FILE_LOG_LEVEL)
        file_formatter = logging.Formatter(
            FILE_FORMAT,
            datefmt=FILE_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)

        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def setup_module_logger(module_name: str, verbose: bool = False) -> logging.Logger:
    """
    Set up a logger for a specific module with optional verbose mode.

    Args:
        module_name: Name of the module
        verbose: If True, sets console output to DEBUG level

    Returns:
        Configured logger instance
    """
    logger = get_logger(module_name)

    if verbose:
        # Set console handler to DEBUG for this logger
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG)

    return logger


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use

    Example:
        >>> logger = get_logger(__name__)
        >>> @log_execution_time(logger)
        ... def slow_function():
        ...     time.sleep(1)
    """
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func.__name__} completed in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {elapsed:.2f} seconds: {str(e)}",
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


def log_dataframe_info(logger: logging.Logger, df, df_name: str = "DataFrame"):
    """
    Log information about a pandas DataFrame.

    Args:
        logger: Logger instance to use
        df: Pandas DataFrame
        df_name: Name to use in log message
    """
    try:
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            logger.warning(f"{df_name} is not a DataFrame: {type(df)}")
            return

        logger.info(
            f"{df_name} info - Shape: {df.shape}, "
            f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB, "
            f"Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}"
        )

        # Log null counts if any
        null_counts = df.isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0]
            logger.debug(f"{df_name} null values: {dict(null_cols)}")

    except Exception as e:
        logger.error(f"Error logging DataFrame info: {str(e)}")


# Create a default logger for this module
module_logger = get_logger(__name__)
module_logger.info("Logging module initialized for new_pipeline")
